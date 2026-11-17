import copy
import difflib
import json
import math
import random
import time
from typing import Any, Optional

from solver.mcts.node import FormulationNode
from solver.utils import compact_for_prompt, llm_usage_context, parse_llm_structured_response_with_meta, render_prompt_template


def _finite_float_or_none(value: Any) -> Optional[float]:
    """Convert a value into a finite float, returning None when conversion is unsafe.
    
    Args:
        value: Arbitrary value to normalize, inspect, parse, or serialize.
    
    Returns:
        The value described in the function summary.
    """
    if value is None:
        return None
    try:
        float_value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(float_value):
        return None
    return float_value


class DynamicMCTS:
    """
    Monte Carlo Tree Search for optimization formulation generation with dynamic component generation
    """

    def __init__(
        self,
        problem_description,
        llm_component_generator,
        evaluation_function,
        exploration_weight=1.0,
        initial_accept_threshold=0.0,
    ):
        """Initialize the object and store the state required by later methods.
        
        Args:
            problem_description: Natural-language optimization problem statement.
            llm_component_generator: Input value used by this function.
            evaluation_function: Input value used by this function.
            exploration_weight: Input value used by this function.
            initial_accept_threshold: Score threshold for accepting the fast initial candidate.
        """
        self.root = FormulationNode()
        self.problem_description = problem_description
        self.llm_component_generator = llm_component_generator
        self.evaluate_formulation = evaluation_function
        self.exploration_weight = exploration_weight
        self.initial_accept_threshold = initial_accept_threshold
        self.suggestions_for_sets = []
        self.suggestions_for_parameters = []
        self.suggestions_for_variables = []
        self.suggestions_for_objective = []
        self.suggestions_for_constraints = []
        self.suggestions_for_code = []
        self.operation_log = []
        self.current_iter = -1
        self.best_score_report = None
        self.best_review = None
        self.initial_simulate_review = None
        self.initial_simulate_check = None
        self.selection_uncertainty_weight = 0.25
        self.objective_pool = []
        self.max_revision_per_node = 2
        self.empty_component_retry_limit = 3
        self.api_retry_sleep_seconds = 3
        self.phase_timing = {
            "fast_seconds": 0.0,
            "slow_seconds": 0.0,
        }

    def _dedupe_suggestions(self, suggestions):
        """De-duplicate accumulated revision suggestions for stable prompting.
        
        Args:
            suggestions: Accumulated natural-language repair suggestions.
        
        Returns:
            The value described in the function summary.
        """
        seen = set()
        unique = []
        for suggestion in suggestions or []:
            text = str(suggestion or "").strip()
            key = " ".join(text.split()).lower()
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(text)
        return unique

    def _append_unique_suggestion(self, suggestions, prompt):
        """Append a suggestion only if its normalized text has not already been stored.
        
        Args:
            suggestions: Accumulated natural-language repair suggestions.
            prompt: Prompt text sent to the LLM endpoint.
        
        Returns:
            The value described in the function summary.
        """
        text = str(prompt or "").strip()
        if not text:
            return False
        key = " ".join(text.split()).lower()
        existing = {" ".join(str(item or "").strip().split()).lower() for item in suggestions}
        if key in existing:
            return False
        suggestions.append(text)
        return True

    def unique_code_suggestions(self):
        """Return de-duplicated code-level suggestions accumulated during search.
        
        Returns:
            The value described in the function summary.
        """
        return self._dedupe_suggestions(self.suggestions_for_code)

    def suggestions_snapshot(self):
        """Return a de-duplicated snapshot of all suggestion pools.
        
        Returns:
            The value described in the function summary.
        """
        return {
            "sets": self._dedupe_suggestions(self.suggestions_for_sets),
            "parameters": self._dedupe_suggestions(self.suggestions_for_parameters),
            "variables": self._dedupe_suggestions(self.suggestions_for_variables),
            "objective": self._dedupe_suggestions(self.suggestions_for_objective),
            "constraints": self._dedupe_suggestions(self.suggestions_for_constraints),
            "code": self.unique_code_suggestions(),
        }

    def raw_suggestions_snapshot(self):
        """Return raw suggestion pools without de-duplication for debugging.
        
        Returns:
            The value described in the function summary.
        """
        return {
            "sets": list(self.suggestions_for_sets),
            "parameters": list(self.suggestions_for_parameters),
            "variables": list(self.suggestions_for_variables),
            "objective": list(self.suggestions_for_objective),
            "constraints": list(self.suggestions_for_constraints),
            "code": list(self.suggestions_for_code),
        }

    def _safe_json(self, value):
        """Serialize diagnostic values for operation-log output without raising on non-JSON objects.
        
        Args:
            value: Arbitrary value to normalize, inspect, parse, or serialize.
        
        Returns:
            The value described in the function summary.
        """
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return str(value)

    def _node_label(self, node):
        """Build a compact text label describing node statistics.
        
        Args:
            node: MCTS tree node being inspected, expanded, or updated.
        
        Returns:
            The value described in the function summary.
        """
        return (
            f"level={node.level}, visits={node.visits}, reward={node.rewards:.4f}, "
            f"expanded={node.expanded}, children={len(node.children)}, "
            f"available={len(node.available_components)}, need_revision={node.need_revision}, "
            f"revision_count={node.revision_count}"
        )

    def _node_path(self, node):
        """Return the level path from the root to a node.
        
        Args:
            node: MCTS tree node being inspected, expanded, or updated.
        
        Returns:
            The value described in the function summary.
        """
        path = []
        current = node
        while current is not None:
            path.append(f"L{current.level}")
            current = current.parent
        return " -> ".join(reversed(path))

    def _node_runtime_id(self, node):
        """Create a deterministic runtime ID for a node based on sibling indices.
        
        Args:
            node: MCTS tree node being inspected, expanded, or updated.
        
        Returns:
            The value described in the function summary.
        """
        if node is self.root or node.parent is None:
            return "root"

        parts = []
        current = node
        while current.parent is not None:
            parent = current.parent
            try:
                child_index = parent.children.index(current)
            except ValueError:
                child_index = 0
            parts.append(str(child_index))
            current = parent
        return "n_" + "_".join(reversed(parts))

    def _node_runtime_chain(self, node):
        """Return the runtime ID chain from root to node.
        
        Args:
            node: MCTS tree node being inspected, expanded, or updated.
        
        Returns:
            The value described in the function summary.
        """
        chain = []
        current = node
        while current is not None:
            chain.append(self._node_runtime_id(current))
            current = current.parent
        chain.reverse()
        return chain

    def _log_event(self, iter_count, step, message, **details):
        """Append a structured event to the MCTS operation log.
        
        Args:
            iter_count: Input value used by this function.
            step: Input value used by this function.
            message: Input value used by this function.
            details: Input value used by this function.
        """
        lines = [f"[iter {iter_count:03d}] {step}: {message}"]
        for key, value in details.items():
            lines.append(f"  - {key}: {self._safe_json(value)}")
        self.operation_log.append("\n".join(lines))

    def operation_log_text(self):
        """Render the accumulated MCTS operation log as markdown text.
        
        Returns:
            The value described in the function summary.
        """
        header = [
            "# MCTS Operation Log",
            "",
            "This file records each search iteration step by step.",
            "",
        ]
        return "\n".join(header + self.operation_log) + ("\n" if self.operation_log else "")

    def _seed_backbone_from_rollout(self):
        """Seed the tree with a complete initial rollout formulation as the backbone path.
        
        Returns:
            The value described in the function summary.
        """
        rollout_formulation = self._generate_complete_initial_formulation_with_fallback()

        sets_component = rollout_formulation.get("sets") or []
        parameters_component = rollout_formulation.get("parameters") or []
        variables_component = rollout_formulation.get("variables") or []
        objective_component = rollout_formulation.get("objective")
        constraints_component = rollout_formulation.get("constraints") or []

        if sets_component is None or not parameters_component or not variables_component or objective_component is None or not constraints_component:
            raise RuntimeError("Initial backbone rollout did not produce a complete formulation.")

        chain_components = [
            sets_component,
            parameters_component,
            variables_component,
            objective_component,
            constraints_component,
        ]

        parent = self.root
        leaf = None
        for level, component in enumerate(chain_components, start=1):
            child = FormulationNode(parent=parent, level=level, component=component)
            parent.children.append(child)
            parent.available_components = [component]
            parent.expanded = False
            leaf = child
            parent = child

        return leaf, rollout_formulation

    def _generate_complete_initial_formulation_with_fallback(self):
        """Try fast full-formulation generation and fall back to layered simulation on failure.
        
        Returns:
            The value described in the function summary.
        """
        try:
            return self._generate_complete_initial_formulation()
        except RuntimeError as exc:
            self._log_event(
                self.current_iter,
                "fast_full_formulation_fallback",
                "Fast full-formulation generation failed; falling back to layered rollout generation",
                error=str(exc),
            )
            return self._simulate(self.root)

    def _generate_complete_initial_formulation(self):
        """Generate and validate an initial complete formulation with retry metadata.
        
        Returns:
            The value described in the function summary.
        """
        last_formulation = {}
        last_failure_meta = {}
        for attempt in range(1, self.empty_component_retry_limit + 1):
            formulation = self.llm_component_generator.generate_full_formulation(self.problem_description)
            last_formulation = formulation or {}
            last_failure_meta = dict(getattr(self.llm_component_generator, "last_call_meta", {}) or {})
            if self._is_complete_formulation(last_formulation):
                return last_formulation

            self._log_event(
                self.current_iter,
                "fast_full_formulation_retry",
                "Fast full-formulation generation did not produce a complete formulation; retrying",
                attempt=attempt,
                retry_limit=self.empty_component_retry_limit,
                api_status=last_failure_meta.get("api_status"),
                http_status=last_failure_meta.get("http_status"),
                parse_status=last_failure_meta.get("parse_status"),
                parse_error=last_failure_meta.get("parse_error"),
                api_error=last_failure_meta.get("error"),
                parser_used=last_failure_meta.get("parser_used"),
                raw_response=last_failure_meta.get("raw_response"),
                cleaned_response=last_failure_meta.get("cleaned_response"),
            )
            if attempt < self.empty_component_retry_limit:
                time.sleep(self.api_retry_sleep_seconds)

        raise RuntimeError(
            "Fast full-formulation generation did not produce a complete formulation "
            f"after {self.empty_component_retry_limit} retries. "
            f"api_status={last_failure_meta.get('api_status')}, "
            f"http_status={last_failure_meta.get('http_status')}, "
            f"parse_status={last_failure_meta.get('parse_status')}, "
            f"parse_error={last_failure_meta.get('parse_error')}, "
            f"api_error={last_failure_meta.get('error')}, "
            f"last_formulation={last_formulation}, "
            f"raw_response={last_failure_meta.get('raw_response')}"
        )

    def _is_complete_formulation(self, formulation):
        """Check whether a formulation contains all required non-empty components.
        
        Args:
            formulation: Structured five-element optimization formulation.
        
        Returns:
            The value described in the function summary.
        """
        return (
            isinstance(formulation, dict)
            and isinstance(formulation.get("sets"), list)
            and bool(formulation.get("parameters"))
            and bool(formulation.get("variables"))
            and formulation.get("objective") is not None
            and bool(formulation.get("constraints"))
        )

    def _initial_accept_gate(self, reward, score_report, reasoning):
        """Decide whether a fast candidate passes the configured reliability threshold.
        
        Args:
            reward: Scalar score assigned to an evaluated formulation.
            score_report: Structured score report containing reward components and evidence.
            reasoning: Diagnostic reasoning or review payload for a candidate formulation.
        
        Returns:
            The value described in the function summary.
        """
        threshold = self.initial_accept_threshold or 0.0
        failed_gates = []
        if threshold <= 0:
            return False, ["initial_accept_disabled"]

        reward_value = _finite_float_or_none(reward)
        if reward_value is None or reward_value < threshold:
            failed_gates.append("reward_below_threshold")
        if reward_value is None or reward_value < 1.0:
            failed_gates.append("reward_not_1")

        score_report = score_report or {}
        score_components = score_report.get("score_components") or {}
        evidence = score_report.get("evidence") or {}

        required_exact_scores = ("exec", "solve", "obj", "def", "str")
        for key in required_exact_scores:
            if score_components.get(key) != 1.0:
                failed_gates.append(f"{key}_score_not_1")

        if evidence.get("solve_status") != "optimal":
            failed_gates.append("solve_status_not_optimal")

        objective_value = _finite_float_or_none(evidence.get("optimal_value"))
        if objective_value is None:
            failed_gates.append("objective_value_not_finite")

        if evidence.get("error_message"):
            failed_gates.append("nonempty_error_message")

        review_score = _finite_float_or_none(score_report.get("review_score"))
        if review_score != 1.0:
            failed_gates.append("review_score_not_1")

        review = reasoning if isinstance(reasoning, dict) else {}
        for layer in ("sets", "parameters", "variables", "objective", "constraints", "code"):
            layer_review = review.get(layer) or {}
            if isinstance(layer_review, dict) and layer_review.get("need_revise") == 1:
                failed_gates.append(f"{layer}_review_needs_revision")

        return not failed_gates, failed_gates

    def _initial_accept_confirmation(self, formulation, score_report, reasoning):
        """Second no-ground-truth LLM check before accepting a single initial rollout."""
        score_report = score_report or {}
        pyomo_code = str(score_report.get("_pyomo_code", "") or "")
        execution_output = str(score_report.get("_execution_output", "") or "")
        compact_score_report = self._compact_score_report(score_report)
        prompt = render_prompt_template(
            "mcts_prompts.yaml",
            "initial_accept_confirmation",
            problem_description=self.problem_description,
            formulation_json=json.dumps(compact_for_prompt(formulation), indent=2, ensure_ascii=False),
            pyomo_code=pyomo_code[:16000],
            execution_output=execution_output[:12000],
            score_report_json=json.dumps(compact_for_prompt(compact_score_report), indent=2, ensure_ascii=False),
            reasoning_json=json.dumps(compact_for_prompt(reasoning), indent=2, ensure_ascii=False),
        )
        response = self.llm_component_generator._call_llm_api(
            prompt,
            request_label="initial_accept_confirmation",
            call_type="initial_accept_confirmation",
            step="evaluate",
        )
        parsed, parse_meta = parse_llm_structured_response_with_meta(response, dict)
        accepted = (
            isinstance(parsed, dict)
            and parsed.get("accept") in (1, True, "1", "true", "True", "YES", "yes")
            and str(parsed.get("confidence", "") or "").strip().lower() == "high"
            and not parsed.get("numeric_coverage_issues")
            and not parsed.get("constraint_coverage_issues")
            and not parsed.get("answer_metric_issues")
        )
        details = parsed if isinstance(parsed, dict) else {}
        details["_parse_status"] = parse_meta.get("status")
        details["_parse_error"] = parse_meta.get("error")
        details["_raw_response"] = response
        return accepted, details

    def _print_initial_accept_decision(self, accepted, failed_gates, reward, score_report, reasoning):
        """Print a concise console summary of the fast-acceptance decision.
        
        Args:
            accepted: Whether the fast candidate passed acceptance checks.
            failed_gates: Names of acceptance gates that failed.
            reward: Scalar score assigned to an evaluated formulation.
            score_report: Structured score report containing reward components and evidence.
            reasoning: Diagnostic reasoning or review payload for a candidate formulation.
        """
        score_report = score_report or {}
        score_components = score_report.get("score_components") or {}
        evidence = score_report.get("evidence") or {}
        critical_review = {}
        review = reasoning if isinstance(reasoning, dict) else {}
        for layer in ("variables", "objective", "constraints"):
            layer_review = review.get(layer) or {}
            critical_review[layer] = (
                layer_review.get("need_revise")
                if isinstance(layer_review, dict)
                else None
            )
        print("[Initial accept gate]")
        print(f"  accepted: {accepted}")
        print(f"  threshold: {self.initial_accept_threshold}")
        print(f"  reward: {reward}")
        print(
            "  score_components: "
            f"exec={score_components.get('exec')}, "
            f"solve={score_components.get('solve')}, "
            f"obj={score_components.get('obj')}, "
            f"def={score_components.get('def')}, "
            f"str={score_components.get('str')}"
        )
        print(
            "  evidence: "
            f"solve_status={evidence.get('solve_status')}, "
            f"optimal_value={evidence.get('optimal_value')}, "
            f"error_message={evidence.get('error_message') or ''}"
        )
        print(f"  review_score: {score_report.get('review_score')}")
        print(f"  critical_review_need_revise: {critical_review}")
        semantic_report = score_report.get("semantic_report") or {}
        if semantic_report:
            print(
                "  semantic_report: "
                f"score={semantic_report.get('score')}, "
                f"risk={semantic_report.get('overall_semantic_risk')}, "
                f"issues={semantic_report.get('issues')}"
            )
        print(f"  failed_gates: {failed_gates}")
        
    def print_tree(self):
        """
        Print the tree starting from the root node.
        """
        self._print_node(self.root, "", -1)

    def tree_markdown(self):
        """Render the current search tree as markdown for saved artifacts.
        
        Returns:
            The value described in the function summary.
        """
        lines = ["# MCTS Tree", ""]
        self._append_node_markdown(self.root, lines, depth=0, index="Root")
        return "\n".join(lines) + "\n"

    def _append_node_markdown(self, node, lines, depth, index):
        """Append one node and its descendants to the tree-markdown buffer.
        
        Args:
            node: MCTS tree node being inspected, expanded, or updated.
            lines: Mutable list of markdown or log lines being assembled.
            depth: Current tree depth used for indentation or traversal.
            index: Sibling index used for deterministic labels.
        """
        indent = "  " * depth
        component = "" if node.component is None else f" component={str(node.component)[:500]}"
        lines.append(
            f"{indent}- **{index}** level={node.level}, visits={node.visits}, "
            f"reward={node.rewards:.4f}, need_revision={node.need_revision}, "
            f"uncertainty={node.uncertainty}{component}"
        )
        for child_index, child in enumerate(node.children):
            self._append_node_markdown(child, lines, depth + 1, f"Node{child_index}")

    def _print_node(self, node, prefix, index):
        """
        Recursively print the tree with node number and its level.
        """
        if index == -1:
            print(f"Root(n={node.need_revision}, v={node.visits}, r={node.rewards:.1f})")
        else:
            # Print the current node
            print(prefix + f"Node{index}(n={node.need_revision}, v={node.visits}, r={node.rewards:.1f})")

        # Recursively print each child node at the next level with indentation
        child_prefix = prefix.replace('-',' ') + "   "  # Indentation for children nodes
        for i, child in enumerate(node.children):
            print(child_prefix + "|")
            self._print_node(child, child_prefix + "|--- ", i)

    def search(self, iterations=100, time_limit=60, num_components=3):
        """
        Perform MCTS search for a specified number of iterations or time limit

        Parameters:
        -----------
        iterations : int
            Maximum number of iterations to perform
        time_limit : int
            Maximum time in seconds to search

        Returns:
        --------
        dict
            The best formulation found
        """
        start_time = time.time()
        fast_start_time = start_time
        iter_count = 0
        best_reward = float('-inf')
        best_formulation = None

        self._log_event(0, "search_start", "Starting MCTS search", iterations=iterations, time_limit=time_limit, num_components=num_components)
        with llm_usage_context(
            phase="fast",
            step="generate",
            operation="rollout",
            operation_detail="initial_backbone_rollout",
            iteration=0,
        ):
            backbone_leaf, backbone_formulation = self._seed_backbone_from_rollout()
        self._log_event(
            0,
            "backbone_seed",
            "Initialized a full backbone chain from one complete rollout",
            terminal_path=self._node_path(backbone_leaf),
            terminal_node_id=self._node_runtime_id(backbone_leaf),
            terminal_node_chain=self._node_runtime_chain(backbone_leaf),
        )
        with llm_usage_context(
            phase="fast",
            step="evaluate",
            operation="candidate_evaluation",
            operation_detail="initial_backbone_evaluation",
            iteration=0,
        ):
            evaluation_result = self.evaluate_formulation(
                backbone_formulation,
                self.problem_description,
                code_suggestions=self.unique_code_suggestions(),
                objective_history=self.objective_pool,
            )
        if len(evaluation_result) == 4:
            reward, reasoning, uncertainty, score_report = evaluation_result
        else:
            reward, reasoning, uncertainty = evaluation_result
            score_report = None
        backbone_leaf.score_report = score_report
        backbone_leaf.score_components = (score_report or {}).get("score_components")
        backbone_leaf.evidence = (score_report or {}).get("evidence")
        backbone_leaf.composite_score = (score_report or {}).get("overall_score")
        backbone_leaf.review = copy.deepcopy(reasoning)
        backbone_leaf.last_reward = reward
        self._update_objective_pool(score_report)
        self._log_event(
            0,
            "backbone_evaluation",
            "Evaluated the seeded terminal backbone",
            reward=reward,
            uncertainty=uncertainty,
            score_components=backbone_leaf.score_components,
            evidence=backbone_leaf.evidence,
        )
        best_reward = reward
        best_formulation = copy.deepcopy(backbone_formulation)
        self.best_score_report = copy.deepcopy(score_report)
        self.best_review = copy.deepcopy(reasoning)
        self.initial_simulate_review = copy.deepcopy(reasoning)
        self._log_event(
            0,
            "best_sample_update",
            "Updated best sampled formulation from seeded terminal backbone",
            best_reward=best_reward,
            path=self._node_path(backbone_leaf),
            node_id=self._node_runtime_id(backbone_leaf),
            node_chain=self._node_runtime_chain(backbone_leaf),
        )
        self._backpropagate(backbone_leaf, reward, reasoning, uncertainty, backbone_formulation, score_report)
        initial_accepted, initial_failed_gates = self._initial_accept_gate(
            reward,
            score_report,
            reasoning,
        )
        if self.initial_accept_threshold and self.initial_accept_threshold > 0:
            self._print_initial_accept_decision(
                initial_accepted,
                initial_failed_gates,
                reward,
                score_report,
                reasoning,
            )
        initial_confirmation_details = None
        if initial_accepted:
            try:
                with llm_usage_context(
                    phase="fast",
                    step="evaluate",
                    operation="initial_accept_confirmation",
                    operation_detail="initial_accept_confirmation",
                    iteration=0,
                ):
                    initial_confirmation_accepted, initial_confirmation_details = self._initial_accept_confirmation(
                        backbone_formulation,
                        score_report,
                        reasoning,
                    )
            except Exception as exc:
                initial_confirmation_accepted = False
                initial_confirmation_details = {
                    "_parse_status": "exception",
                    "_parse_error": str(exc),
                    "accept": 0,
                    "confidence": "Low",
                    "blocking_layer": "review",
                    "blocking_issue": "initial accept confirmation failed",
                    "reason": str(exc),
                }
            self.initial_simulate_check = copy.deepcopy(initial_confirmation_details)
            self._log_event(
                0,
                "initial_accept_confirmation",
                "Ran no-ground-truth confirmation before initial acceptance",
                accepted=initial_confirmation_accepted,
                details=initial_confirmation_details,
                path=self._node_path(backbone_leaf),
                node_id=self._node_runtime_id(backbone_leaf),
                node_chain=self._node_runtime_chain(backbone_leaf),
            )
            if not initial_confirmation_accepted:
                initial_accepted = False
                initial_failed_gates = list(initial_failed_gates or []) + ["initial_accept_confirmation_reject"]
        else:
            self.initial_simulate_check = {
                "skipped": True,
                "reason": "initial_accept_gate_failed",
                "failed_gates": list(initial_failed_gates or []),
                "accept": 0,
                "confidence": "Low",
            }
        if initial_accepted:
            self.phase_timing["fast_seconds"] = time.time() - fast_start_time
            self.phase_timing["slow_seconds"] = 0.0
            self._log_event(
                0,
                "initial_accept",
                "Accepted seeded terminal backbone before MCTS iterations",
                reward=reward,
                threshold=self.initial_accept_threshold,
                failed_gates=initial_failed_gates,
                confirmation=initial_confirmation_details,
                path=self._node_path(backbone_leaf),
                node_id=self._node_runtime_id(backbone_leaf),
                node_chain=self._node_runtime_chain(backbone_leaf),
                elapsed_seconds=time.time() - start_time,
            )
            self._log_event(
                0,
                "search_end",
                "Finished MCTS search via initial acceptance",
                completed_iterations=0,
                best_reward=best_reward,
                elapsed_seconds=time.time() - start_time,
            )
            self.current_iter = -1
            return best_formulation
        self.phase_timing["fast_seconds"] = time.time() - fast_start_time
        if self.initial_accept_threshold and self.initial_accept_threshold > 0:
            self._log_event(
                0,
                "initial_accept_reject",
                "Seeded terminal backbone did not pass initial acceptance gate",
                reward=reward,
                threshold=self.initial_accept_threshold,
                failed_gates=initial_failed_gates,
                confirmation=initial_confirmation_details,
                path=self._node_path(backbone_leaf),
                node_id=self._node_runtime_id(backbone_leaf),
                node_chain=self._node_runtime_chain(backbone_leaf),
            )
        slow_start_time = time.time()
        while iter_count < iterations and time.time() - start_time < time_limit:
            self.current_iter = iter_count
            # Selection phase
            forced_revision_node = self._select_pending_revision_node()
            if forced_revision_node is not None:
                node = forced_revision_node
                selection_policy = "forced_revision"
            else:
                node = self._select()
                selection_policy = "ucb"
            self._log_event(
                iter_count,
                "selection",
                "Selected node for expansion/simulation",
                selection_policy=selection_policy,
                node=self._node_label(node),
                path=self._node_path(node),
                node_id=self._node_runtime_id(node),
                node_chain=self._node_runtime_chain(node),
                is_terminal=node.is_terminal(),
                is_fully_expanded=node.is_fully_expanded(),
            )
            self.print_tree()
            # Expansion phase - expand the node if not fully expanded
            if node.need_revision == 1:
                self._log_event(iter_count, "revision_check", "Node requested revision handling", node=self._node_label(node), uncertainty=node.uncertainty)

                # Generate revised components using the local diagnostic context.
                with llm_usage_context(
                    phase="slow",
                    step="generate",
                    operation="revision",
                    operation_detail="revision_expansion",
                    iteration=iter_count,
                ):
                    self._expand_revision_node(node, num_components=num_components)
                self._log_event(
                    iter_count,
                    "revision_expand",
                    "Expanded node due to revision suggestion",
                    node=self._node_label(node),
                    path=self._node_path(node),
                    node_id=self._node_runtime_id(node),
                    node_chain=self._node_runtime_chain(node),
                )

                node.need_revision = 0 # Have addressed revision recommendation
                
                # If node has unexpanded components, create a child node, and step into child
                if not (len(node.children) >= node.max_child_num or len(node.children) == len(node.available_components)):
                    node = self._create_child(node)
                else:
                    # In current situation, the available_components and node.chindren will not change, just continue
                    iter_count += 1
                    continue
                
            else:
                if not node.is_terminal():
                    # Generate components if the node hasn't been expanded yet
                    if not node.expanded:
                        with llm_usage_context(
                            phase="slow",
                            step="generate",
                            operation="expansion",
                            operation_detail="tree_expansion",
                            iteration=iter_count,
                        ):
                            self._expand_node(node, num_components=num_components)
                        self._log_event(
                            iter_count,
                            "expand",
                            "Expanded node because it had not been expanded yet",
                            node=self._node_label(node),
                            path=self._node_path(node),
                            node_id=self._node_runtime_id(node),
                            node_chain=self._node_runtime_chain(node),
                            available_components=len(node.available_components),
                        )

                    # If node has unexpanded components, create a child node, and step into child
                    if not node.is_fully_expanded():
                        parent = node
                        node = self._create_child(node)
                        self._log_event(
                            iter_count,
                            "create_child",
                            "Moved to child node",
                            parent=self._node_label(parent),
                            parent_node_id=self._node_runtime_id(parent),
                            child=self._node_label(node),
                            child_node_id=self._node_runtime_id(node),
                            child_node_chain=self._node_runtime_chain(node),
                            child_path=self._node_path(node),
                            child_component=node.component,
                        )
                    else:
                        self._log_event(
                            iter_count,
                            "no_child_created",
                            "Node was treated as fully expanded",
                            node=self._node_label(node),
                            path=self._node_path(node),
                            node_id=self._node_runtime_id(node),
                            node_chain=self._node_runtime_chain(node),
                        )
            

            # Simulation phase
            if not node.is_terminal():
                self._log_event(
                    iter_count,
                    "simulate_start",
                    "Current node is non-terminal; completing remaining levels",
                    node=self._node_label(node),
                    path=self._node_path(node),
                    node_id=self._node_runtime_id(node),
                    node_chain=self._node_runtime_chain(node),
                    current_level=node.level,
                )
                with llm_usage_context(
                    phase="slow",
                    step="generate",
                    operation="rollout",
                    operation_detail="simulation_rollout",
                    iteration=iter_count,
                ):
                    formulation = self._simulate(node)
                self._log_event(iter_count, "simulate_end", "Rollout produced a formulation", completed_keys={"sets": len(formulation['sets']), "parameters": len(formulation['parameters']), "variables": len(formulation['variables']), "has_objective": formulation['objective'] is not None, "constraints": len(formulation['constraints'])})
            else:
                formulation = node.formulation
                self._log_event(
                    iter_count,
                    "terminal_use",
                    "Current node is terminal; evaluating directly",
                    node=self._node_label(node),
                    path=self._node_path(node),
                    node_id=self._node_runtime_id(node),
                    node_chain=self._node_runtime_chain(node),
                )
            
            with llm_usage_context(
                phase="slow",
                step="evaluate",
                operation="candidate_evaluation",
                operation_detail="mcts_candidate_evaluation",
                iteration=iter_count,
            ):
                evaluation_result = self.evaluate_formulation(
                    formulation,
                    self.problem_description,
                    code_suggestions=self.unique_code_suggestions(),
                    objective_history=self.objective_pool,
                )
            if len(evaluation_result) == 4:
                reward, reasoning, uncertainty, score_report = evaluation_result
            else:
                reward, reasoning, uncertainty = evaluation_result
                score_report = None
            new_formulation = formulation
            node.score_report = score_report
            node.score_components = (score_report or {}).get("score_components")
            node.evidence = (score_report or {}).get("evidence")
            node.composite_score = (score_report or {}).get("overall_score")
            node.review = copy.deepcopy(reasoning)
            node.last_reward = reward
            self._update_objective_pool(score_report)
            self._log_event(iter_count, "evaluation", "Evaluated formulation", reward=reward, uncertainty=uncertainty, score_components=node.score_components, evidence=node.evidence)

            # Keep the best single sampled terminal result for diagnostics only.
            # Final selection below uses MCTS terminal-node statistics after search ends.
            if node.is_terminal() and reward > best_reward:
                best_reward = reward
                best_formulation = copy.deepcopy(new_formulation)
                self.best_score_report = copy.deepcopy(score_report)
                self.best_review = copy.deepcopy(reasoning)
                self._log_event(
                    iter_count,
                    "best_sample_update",
                    "Updated best sampled formulation",
                    best_reward=best_reward,
                    path=self._node_path(node),
                    node_id=self._node_runtime_id(node),
                    node_chain=self._node_runtime_chain(node),
                )

            # Backpropagation phase
            self._backpropagate(node, reward, reasoning, uncertainty, formulation, score_report)
            self._log_event(iter_count, "backpropagation", "Backpropagated reward through selected path", reward=reward, path=self._node_path(node), root_reward=self.root.rewards, root_visits=self.root.visits, suggestions=self.suggestions_snapshot())

            iter_count += 1

        self.phase_timing["slow_seconds"] = time.time() - slow_start_time

        # Return the best terminal path after all iterations, using MCTS statistics
        # instead of the highest single sampled reward.
        final_node = self._get_best_formulation()
        best_formulation = copy.deepcopy(final_node.formulation)
        self.best_score_report = copy.deepcopy(final_node.score_report)
        self.best_review = copy.deepcopy(final_node.review)
        self._log_event(
            iter_count,
            "final_selection",
            "Selected final formulation from terminal tree nodes using MCTS statistics",
            path=self._node_path(final_node),
            node_id=self._node_runtime_id(final_node),
            node_chain=self._node_runtime_chain(final_node),
            visits=final_node.visits,
            mean_reward=final_node.rewards,
            mean_uncertainty=final_node.mean_uncertainty,
            last_reward=final_node.last_reward,
            best_sample_reward=best_reward,
        )

        self._log_event(iter_count, "search_end", "Finished MCTS search", completed_iterations=iter_count, best_reward=best_reward, elapsed_seconds=time.time() - start_time)
        self.current_iter = -1
        return best_formulation

    def _update_objective_pool(self, score_report):
        """Store objective values from evaluated candidates for consensus-based scoring.
        
        Args:
            score_report: Structured score report containing reward components and evidence.
        """
        evidence = (score_report or {}).get("evidence") or {}
        objective_value = _finite_float_or_none(evidence.get("optimal_value"))
        solve_status = evidence.get("solve_status")
        if objective_value is None:
            return
        if solve_status not in {"optimal", "feasible", "time_limit", "unknown"}:
            return
        self.objective_pool.append(objective_value)

    def _select(self):
        """
        Select a node to expand using an uncertainty-aware UCB rule

        Returns:
        --------
        FormulationNode
            The selected node
        """
        node = self.root

        # Traverse the tree until we reach a leaf node or non-fully expanded node
        while not node.is_terminal() and node.is_fully_expanded():

            # Find child with highest selection score
            best_child = None
            best_score = float('-inf')

            for child in node.children:
                selection_score = child.get_selection_score(
                    node.visits,
                    self.exploration_weight,
                    self.selection_uncertainty_weight,
                )
                if selection_score > best_score:
                    best_score = selection_score
                    best_child = child

            if best_child is None:
                break

            node = best_child

        return node

    def _pending_revision_requests_for_next_level(self, node):
        """Return pending revision requests that target the child level of a node.
        
        Args:
            node: MCTS tree node being inspected, expanded, or updated.
        
        Returns:
            The value described in the function summary.
        """
        next_level = node.level + 1
        return [
            request for request in getattr(node, "revision_requests", [])
            if request.get("target_level") == next_level
        ]

    def _select_pending_revision_node(self):
        """Prioritize concrete pending revisions before ordinary UCB selection."""
        candidates = []
        for node in self._get_all_nodes(self.root):
            if node.is_terminal():
                continue
            if node.need_revision != 1:
                continue
            if node.revision_count > self.max_revision_per_node:
                continue
            pending_requests = self._pending_revision_requests_for_next_level(node)
            if not pending_requests:
                continue
            candidates.append(node)

        if not candidates:
            return None

        # Shallow revisions repair upstream modeling choices before spending budget deeper.
        return min(
            candidates,
            key=lambda item: (
                item.level,
                item.revision_count,
                -float(getattr(item, "uncertainty", 0.0) or 0.0),
                self._node_path(item),
            ),
        )

    def _expand_node(self, node, num_components=3, similarity_threshold=0.8):
        """
        Generate multiple component options for a node and remove similar formulations

        Parameters:
        -----------
        node : FormulationNode
            The node to expand
        num_components : int
            Number of components to generate from LLM (H in the description)
        similarity_threshold : float
            Threshold for determining if two formulations are too similar (0-1)
        """
        # Determine what level of components to generate
        next_level = node.level + 1

        # Generate H components for the next level
        all_components = []
        for i in range(num_components):
            components = self._generate_nonempty_components(
                node.formulation,
                next_level,
                context_label=f"expand@{self._node_path(node)} batch {i + 1}/{num_components}",
            )
            if components:
                all_components.append(components)

        node.available_components += all_components
        # Remove similar formulations to reduce redundancy
        unique_components = self._filter_similar_formulations(node.available_components, similarity_threshold)

        # Store the unique components in the node
        node.available_components = unique_components
        node.expanded = True

    def _expand_revision_node(self, node, num_components=3, similarity_threshold=0.8):
        """Generate replacement components from node-local revision diagnostics."""
        next_level = node.level + 1
        revision_requests = [
            request for request in getattr(node, "revision_requests", [])
            if request.get("target_level") == next_level
        ]

        all_components = []
        for i in range(num_components):
            components = self._generate_nonempty_revision_components(
                node.formulation,
                next_level,
                revision_requests,
                context_label=f"revise@{self._node_path(node)} batch {i + 1}/{num_components}",
            )
            if components:
                all_components.append(components)

        node.available_components += all_components
        node.available_components = self._filter_similar_formulations(node.available_components, similarity_threshold)
        node.expanded = True
        node.revision_requests = [
            request for request in getattr(node, "revision_requests", [])
            if request.get("target_level") != next_level
        ]


    def _filter_similar_formulations(self, components, similarity_threshold):
        """
        Filter out components that are too similar to others

        Parameters:
        -----------
        components : list
            List of formulation components to filter
        similarity_threshold : float
            Threshold for determining if two formulations are too similar (0-1)

        Returns:
        --------
        list
            Filtered list of components with reduced redundancy
        """
        if not components:
            return []

        def similarity(a, b):
            """Calculate string similarity ratio between two formulations"""
            return difflib.SequenceMatcher(None, str(a), str(b)).ratio()

        # Keep track of components we want to keep
        unique_components = [components[0]]

        # Compare each component with the ones we've decided to keep
        for component in components[1:]:
            is_unique = True
            for unique_comp in unique_components:
                # Compare the formulation strings
                if similarity(component, unique_comp) > similarity_threshold:
                    is_unique = False
                    break

            if is_unique:
                unique_components.append(component)

        return unique_components

    def _create_child(self, node):
        """
        Create a child node by selecting an unexpanded component

        Parameters:
        -----------
        node : FormulationNode
            The parent node

        Returns:
        --------
        FormulationNode
            The new child node
        """
        # Get unexpanded components
        unexpanded_components = node.get_unexpanded_components()

        if not unexpanded_components:
            return node

        next_level = node.level + 1
        component = random.choice(unexpanded_components)

        # Create a new child node
        child = FormulationNode(parent=node, level=next_level, component=component)


        node.children.append(child)
        return child

    def _generate_nonempty_components(self, formulation, level, context_label):
        """Generate non-empty component candidates with retries and diagnostic error reporting.
        
        Args:
            formulation: Structured five-element optimization formulation.
            level: Formulation layer index in the MCTS tree.
            context_label: Human-readable label used in retry/error diagnostics.
        
        Returns:
            The value described in the function summary.
        """
        last_components = []
        last_failure_meta = {}
        attempt = 0
        while True:
            attempt += 1
            components = self.llm_component_generator.generate_next_components(
                self.problem_description, formulation, self, level
            )
            # Empty sets list is valid for problems with no indexed sets.
            if components or level == FormulationNode.SETS_LEVEL:
                return components
            last_components = components or []
            last_failure_meta = dict(getattr(self.llm_component_generator, "last_call_meta", {}) or {})
            api_status = last_failure_meta.get("api_status", "unknown")
            parse_status = last_failure_meta.get("parse_status", "unknown")
            if api_status != "success":
                failure_kind = "api_failure"
            elif parse_status not in ("success", None):
                failure_kind = "parse_failure"
            else:
                failure_kind = "llm_empty_list"
            raw_response = last_failure_meta.get("raw_response")
            retry_until_parse_success = failure_kind == "parse_failure" and bool(raw_response)
            if not retry_until_parse_success and attempt >= self.empty_component_retry_limit:
                break
            self._log_event(
                self.current_iter,
                "empty_component_retry",
                "LLM component generation returned no usable components; retrying",
                level=level,
                attempt=attempt,
                retry_limit="until_parse_success" if retry_until_parse_success else self.empty_component_retry_limit,
                context=context_label,
                failure_kind=failure_kind,
                api_status=api_status,
                http_status=last_failure_meta.get("http_status"),
                parse_status=parse_status,
                parse_error=last_failure_meta.get("parse_error"),
                api_error=last_failure_meta.get("error"),
                parser_used=last_failure_meta.get("parser_used"),
                raw_response=last_failure_meta.get("raw_response"),
                cleaned_response=last_failure_meta.get("cleaned_response"),
                prompt_preview=last_failure_meta.get("prompt_preview"),
            )
            time.sleep(self.api_retry_sleep_seconds)

        api_status = last_failure_meta.get("api_status", "unknown")
        parse_status = last_failure_meta.get("parse_status", "unknown")
        if api_status != "success":
            failure_kind = "api_failure"
        elif parse_status not in ("success", None):
            failure_kind = "parse_failure"
        else:
            failure_kind = "llm_empty_list"
        raise RuntimeError(
            f"LLM returned empty components after {self.empty_component_retry_limit} retries "
            f"for level {level} during {context_label}. "
            f"failure_kind={failure_kind}, api_status={api_status}, "
            f"http_status={last_failure_meta.get('http_status')}, parse_status={parse_status}, "
            f"parse_error={last_failure_meta.get('parse_error')}, api_error={last_failure_meta.get('error')}, "
            f"last_response={last_components}, raw_response={last_failure_meta.get('raw_response')}"
        )

    def _generate_nonempty_revision_components(self, formulation, level, revision_requests, context_label):
        """Generate non-empty revision candidates with retries and diagnostic error reporting.
        
        Args:
            formulation: Structured five-element optimization formulation.
            level: Formulation layer index in the MCTS tree.
            revision_requests: Pending diagnostic requests for component-level repair.
            context_label: Human-readable label used in retry/error diagnostics.
        
        Returns:
            The value described in the function summary.
        """
        if not revision_requests:
            return self._generate_nonempty_components(formulation, level, context_label)

        last_components = []
        last_failure_meta = {}
        attempt = 0
        while True:
            attempt += 1
            components = self.llm_component_generator.refine_next_components(
                self.problem_description,
                formulation,
                self,
                level,
                revision_requests,
            )
            # Empty sets list is valid for problems with no indexed sets.
            if components or level == FormulationNode.SETS_LEVEL:
                return components
            last_components = components or []
            last_failure_meta = dict(getattr(self.llm_component_generator, "last_call_meta", {}) or {})
            api_status = last_failure_meta.get("api_status", "unknown")
            parse_status = last_failure_meta.get("parse_status", "unknown")
            if api_status != "success":
                failure_kind = "api_failure"
            elif parse_status not in ("success", None):
                failure_kind = "parse_failure"
            else:
                failure_kind = "llm_empty_list"
            raw_response = last_failure_meta.get("raw_response")
            retry_until_parse_success = failure_kind == "parse_failure" and bool(raw_response)
            if not retry_until_parse_success and attempt >= self.empty_component_retry_limit:
                break
            self._log_event(
                self.current_iter,
                "empty_revision_retry",
                "LLM revision component generation returned no usable components; retrying",
                level=level,
                attempt=attempt,
                retry_limit="until_parse_success" if retry_until_parse_success else self.empty_component_retry_limit,
                context=context_label,
                request_count=len(revision_requests),
                failure_kind=failure_kind,
                api_status=api_status,
                http_status=last_failure_meta.get("http_status"),
                parse_status=parse_status,
                parse_error=last_failure_meta.get("parse_error"),
                api_error=last_failure_meta.get("error"),
                parser_used=last_failure_meta.get("parser_used"),
                raw_response=last_failure_meta.get("raw_response"),
                cleaned_response=last_failure_meta.get("cleaned_response"),
                prompt_preview=last_failure_meta.get("prompt_preview"),
            )
            time.sleep(self.api_retry_sleep_seconds)

        api_status = last_failure_meta.get("api_status", "unknown")
        parse_status = last_failure_meta.get("parse_status", "unknown")
        if api_status != "success":
            failure_kind = "api_failure"
        elif parse_status not in ("success", None):
            failure_kind = "parse_failure"
        else:
            failure_kind = "llm_empty_list"
        raise RuntimeError(
            f"LLM returned empty revised components after {self.empty_component_retry_limit} retries "
            f"for level {level} during {context_label}. "
            f"failure_kind={failure_kind}, request_count={len(revision_requests)}, "
            f"api_status={api_status}, http_status={last_failure_meta.get('http_status')}, "
            f"parse_status={parse_status}, parse_error={last_failure_meta.get('parse_error')}, "
            f"api_error={last_failure_meta.get('error')}, parser_used={last_failure_meta.get('parser_used')}, "
            f"raw_response={last_failure_meta.get('raw_response')}, last_response={last_components}"
        )

    def _simulate(self, node):
        """
        Simulate by completing the formulation from the current node

        Parameters:
        -----------
        node : FormulationNode
            The starting node for simulation

        Returns:
        --------
        dict
            A complete formulation
        """
        formulation = copy.deepcopy(node.formulation)
        current_level = node.level

        # Complete the formulation by generating components for each remaining level
        while current_level < FormulationNode.FINAL_LEVEL:
            current_level += 1

            # Generate components for this level if needed
            components = self._generate_nonempty_components(
                formulation,
                current_level,
                context_label=f"simulate@{self._node_path(node)} level {current_level}",
            )
            self._log_event(self.current_iter, "simulate_level", "Generated rollout components for a missing level", start_node_path=self._node_path(node), rollout_level=current_level, component_count=len(components), components=components)

            # Add components to the formulation
            if current_level == FormulationNode.SETS_LEVEL:  # Sets
                if not formulation['sets'] and components:
                    # Add all generated sets (usually there's just one set for most problems)
                    formulation['sets'].extend(components)

            elif current_level == FormulationNode.PARAMETERS_LEVEL:  # Parameters
                if not formulation['parameters'] and components:
                    # Add all generated parameters
                    formulation['parameters'].extend(components)

            elif current_level == FormulationNode.VARIABLES_LEVEL:  # Variables
                if not formulation['variables'] and components:
                    # Add all generated variables
                    formulation['variables'].extend(components)

            elif current_level == FormulationNode.OBJECTIVE_LEVEL:  # Objective
                if formulation['objective'] is None and components:
                    # Set the objective (just use the first one if multiple are generated)
                    formulation['objective'] = components[0] if components else None

            elif current_level == FormulationNode.CONSTRAINTS_LEVEL:  # Constraints
                if not formulation['constraints'] and components:
                    # Add all generated constraints
                    formulation['constraints'].extend(components)

        return formulation

    def _layer_name_for_level(self, level):
        """Map a numeric tree level to the corresponding suggestion pool name.
        
        Args:
            level: Formulation layer index in the MCTS tree.
        
        Returns:
            The value described in the function summary.
        """
        return {
            FormulationNode.SETS_LEVEL: "sets",
            FormulationNode.PARAMETERS_LEVEL: "parameters",
            FormulationNode.VARIABLES_LEVEL: "variables",
            FormulationNode.OBJECTIVE_LEVEL: "objective",
            FormulationNode.CONSTRAINTS_LEVEL: "constraints",
        }.get(level, "unknown")

    def _target_level_for_parent_level(self, parent_level):
        """Return the formulation level that follows a parent node level.
        
        Args:
            parent_level: Formulation layer index in the MCTS tree.
        
        Returns:
            The value described in the function summary.
        """
        return parent_level + 1

    def _bad_component_for_level(self, formulation, target_level):
        """Return the formulation component associated with a diagnosed faulty level.
        
        Args:
            formulation: Structured five-element optimization formulation.
            target_level: Formulation layer index in the MCTS tree.
        
        Returns:
            The value described in the function summary.
        """
        formulation = formulation or {}
        if target_level == FormulationNode.SETS_LEVEL:
            return copy.deepcopy(formulation.get("sets") or [])
        if target_level == FormulationNode.PARAMETERS_LEVEL:
            return copy.deepcopy(formulation.get("parameters") or [])
        if target_level == FormulationNode.VARIABLES_LEVEL:
            return copy.deepcopy(formulation.get("variables") or [])
        if target_level == FormulationNode.OBJECTIVE_LEVEL:
            return copy.deepcopy(formulation.get("objective"))
        if target_level == FormulationNode.CONSTRAINTS_LEVEL:
            return copy.deepcopy(formulation.get("constraints") or [])
        return None

    def _compact_score_report(self, score_report):
        """Trim score reports before placing them into prompts or logs.
        
        Args:
            score_report: Structured score report containing reward components and evidence.
        
        Returns:
            The value described in the function summary.
        """
        score_report = score_report or {}
        return {
            "overall_score": score_report.get("overall_score"),
            "score_components": score_report.get("score_components"),
            "evidence": score_report.get("evidence"),
            "primary_failure_mode": score_report.get("primary_failure_mode"),
            "actionable_fix_list": score_report.get("actionable_fix_list"),
            "semantic_report": score_report.get("semantic_report"),
            "selection_reason": score_report.get("selection_reason"),
        }

    def _revision_request_key(self, request):
        """Create a stable de-duplication key for a diagnostic revision request.
        
        Args:
            request: Input value used by this function.
        
        Returns:
            The value described in the function summary.
        """
        if not isinstance(request, dict):
            return None
        return (
            request.get("target_level"),
            str(request.get("error_type", "") or "").strip().lower(),
            " ".join(str(request.get("suggestion", "") or "").split()).lower(),
            self._safe_json(request.get("bad_component")),
        )

    def _find_revision_parent(self, path_nodes, target_level):
        """Find the trusted-prefix node that should own a revision request.
        
        Args:
            path_nodes: Input value used by this function.
            target_level: Formulation layer index in the MCTS tree.
        
        Returns:
            The value described in the function summary.
        """
        for candidate in path_nodes:
            if candidate.level == target_level - 1:
                return candidate
        eligible = [candidate for candidate in path_nodes if candidate.level < target_level]
        return max(eligible, key=lambda item: item.level) if eligible else None

    def _register_revision_requests(self, path_nodes, reasoning, evaluated_formulation, score_report):
        """Register component-level diagnostic revision requests along an evaluated path.
        
        Args:
            path_nodes: Input value used by this function.
            reasoning: Diagnostic reasoning or review payload for a candidate formulation.
            evaluated_formulation: Structured five-element optimization formulation.
            score_report: Structured score report containing reward components and evidence.
        """
        layer_specs = [
            (FormulationNode.SETS_LEVEL, "sets", self.suggestions_for_sets),
            (FormulationNode.PARAMETERS_LEVEL, "parameters", self.suggestions_for_parameters),
            (FormulationNode.VARIABLES_LEVEL, "variables", self.suggestions_for_variables),
            (FormulationNode.OBJECTIVE_LEVEL, "objective", self.suggestions_for_objective),
            (FormulationNode.CONSTRAINTS_LEVEL, "constraints", self.suggestions_for_constraints),
        ]
        for target_level, layer_name, suggestions_store in layer_specs:
            suggestion = reasoning.get(layer_name, {}) if isinstance(reasoning, dict) else {}
            if not isinstance(suggestion, dict) or suggestion.get("need_revise") != 1:
                continue
            parent = self._find_revision_parent(path_nodes, target_level)
            if parent is None:
                continue
            self._apply_revision_suggestion(
                parent,
                suggestion,
                suggestions_store,
                target_level=target_level,
                bad_formulation=evaluated_formulation,
                reasoning=reasoning,
                score_report=score_report,
            )

    def _apply_revision_suggestion(self, node, suggestion, suggestions_store, target_level=None, bad_formulation=None, reasoning=None, score_report=None):
        """Store a revision suggestion in the appropriate node and global suggestion pool.
        
        Args:
            node: MCTS tree node being inspected, expanded, or updated.
            suggestion: One natural-language repair suggestion to store or apply.
            suggestions_store: Mutable list that stores accumulated repair suggestions.
            target_level: Formulation layer index in the MCTS tree.
            bad_formulation: Structured five-element optimization formulation.
            reasoning: Diagnostic reasoning or review payload for a candidate formulation.
            score_report: Structured score report containing reward components and evidence.
        """
        if not isinstance(suggestion, dict) or suggestion.get("need_revise") != 1:
            node.need_revision = 0
            return

        if node.revision_count >= self.max_revision_per_node:
            node.need_revision = 0
            return

        self._append_unique_suggestion(suggestions_store, suggestion.get("prompt", ""))
        target_level = target_level or self._target_level_for_parent_level(node.level)
        request = {
            "target_level": target_level,
            "target_layer": self._layer_name_for_level(target_level),
            "bad_component": self._bad_component_for_level(bad_formulation, target_level),
            "bad_formulation": copy.deepcopy(bad_formulation),
            "prefix_formulation": copy.deepcopy(node.formulation),
            "review_item": copy.deepcopy(suggestion),
            "error_type": str(suggestion.get("error_type", "") or "unknown"),
            "reason": str(suggestion.get("reason", "") or ""),
            "suggestion": str(suggestion.get("prompt", "") or ""),
            "semantic_checks": copy.deepcopy((reasoning or {}).get("semantic_checks", {})),
            "score_report": self._compact_score_report(score_report),
        }
        request_key = self._revision_request_key(request)
        existing_keys = {
            self._revision_request_key(existing)
            for existing in getattr(node, "revision_requests", [])
        }
        if request_key in existing_keys:
            node.mean_uncertainty = max(node.mean_uncertainty, suggestion.get("uncertainty", 0))
            node.uncertainty = node.mean_uncertainty
            node.need_revision = 1
            return
        node.revision_requests.append(request)
        node.mean_uncertainty = max(node.mean_uncertainty, suggestion.get("uncertainty", 0))
        node.uncertainty = node.mean_uncertainty
        node.need_revision = 1
        node.revision_count += 1

    def _backpropagate(self, node, reward, reasoning, uncertainty, evaluated_formulation=None, score_report=None):
        """
        Update statistics for all nodes in the path from node to root

        Parameters:
        -----------
        node : FormulationNode
            The node to start backpropagation from
        reward : float
            The reward value to backpropagate
        reasoning: json
            Revision suggestions from LLM reasoning
        """
        reasoning = reasoning if isinstance(reasoning, dict) else {}
        code_suggestion = reasoning.get("code", {})
        if isinstance(code_suggestion, dict) and code_suggestion.get("need_revise") == 1:
            self._append_unique_suggestion(self.suggestions_for_code, code_suggestion.get("prompt", ""))

        path_nodes = []
        current = node
        while current is not None:
            path_nodes.append(current)
            current = current.parent
        self._register_revision_requests(path_nodes, reasoning, evaluated_formulation, score_report)

        rho = math.exp(-max(0.0, uncertainty))
        while node is not None:
            prev_mean = node.rewards
            node.visits += 1

            # Uncertainty-weighted average reward, following SolverLLM's backpropagation form.
            node.rewards = node.rewards + rho * (reward - node.rewards) / node.visits
            node.reward_m2 += (reward - prev_mean) * (reward - node.rewards)
            node.mean_uncertainty = node.mean_uncertainty + (uncertainty - node.mean_uncertainty) / node.visits
            node.uncertainty = node.mean_uncertainty

            node = node.parent

    def _get_best_formulation(self):
        """
        Select the final formulation from terminal tree nodes using MCTS statistics.

        Returns:
        --------
        FormulationNode
            The node with the best formulation
        """
        # Final answer must come from a real terminal tree path.
        terminal_nodes = self._get_terminal_nodes(self.root)

        if not terminal_nodes:
            raise RuntimeError("No terminal formulation exists in the tree; final selection requires a complete terminal path.")

        return max(
            terminal_nodes,
            key=lambda node: (
                node.rewards,
                node.visits,
                -node.mean_uncertainty,
                _finite_float_or_none(node.last_reward) or float("-inf"),
            ),
        )

    def _get_terminal_nodes(self, node):
        """
        Get all terminal nodes in the subtree rooted at node

        Parameters:
        -----------
        node : FormulationNode
            The root of the subtree

        Returns:
        --------
        list
            Terminal nodes in the subtree
        """
        terminal_nodes = []

        if node.is_terminal():
            terminal_nodes.append(node)

        for child in node.children:
            terminal_nodes.extend(self._get_terminal_nodes(child))

        return terminal_nodes

    def _get_all_nodes(self, node):
        """
        Get all nodes in the subtree rooted at node

        Parameters:
        -----------
        node : FormulationNode
            The root of the subtree

        Returns:
        --------
        list
            All nodes in the subtree
        """
        all_nodes = [node]

        for child in node.children:
            all_nodes.extend(self._get_all_nodes(child))

        return all_nodes

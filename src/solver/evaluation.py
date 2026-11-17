from typing import Any, Tuple, Optional
import json
import os
import re
import requests
import math
import statistics
import time
from solver.utils import compact_for_prompt, normalize_for_json, record_llm_call, render_prompt_template

LLM_REQUEST_TIMEOUT = 120
GUROBI_TIME_LIMIT_SECONDS = 300


SCORE_KEYS = ("exec", "obj", "def", "str", "solve")
SELECTION_WEIGHTS = {
    "exec": 0.3,
    "obj": 0.4,
    "def": 0.1,
    "str": 0.1,
    "solve": 0.1,
}

REVIEW_KEYS = ("sets", "parameters", "variables", "objective", "constraints", "code")
REVIEW_PENALTY_WEIGHTS = {
    "sets": 0.10,
    "parameters": 0.15,
    "variables": 0.20,
    "objective": 0.25,
    "constraints": 0.25,
    "code": 0.15,
}

CONSENSUS_MIN_POOL = 4
CONSENSUS_MAX_WEIGHT = 0.15
CONSENSUS_WARMUP_POOL = 8
CONSENSUS_REL_TOL = 1e-4


def _chat_completions_url():
    """Build the chat-completions endpoint URL from the configured OpenAI-compatible base URL.
    
    Returns:
        The value described in the function summary.
    """
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    return f"{base_url}/chat/completions"


def _llm_model():
    """Return the model name used for evaluation and code-generation LLM calls.
    
    Returns:
        The value described in the function summary.
    """
    return os.environ.get("OPENAI_MODEL", "gpt-4o")


def empty_review():
    """Return the default structured review object used when LLM review is unavailable or fails.
    
    Returns:
        The value described in the function summary.
    """
    return {
        "review_summary": {
            "definition_consistency": "",
            "structural_soundness": "",
            "pyomo_faithfulness": "",
            "numerical_validity": "",
            "primary_issue": "",
        },
        "diagnosis": {
            "definition_consistency": {"score": 0.0, "issues": [], "confidence": "Low"},
            "structural_soundness": {"score": 0.0, "issues": [], "confidence": "Low"},
            "implementation_faithfulness": {"score": 0.0, "issues": [], "confidence": "Low"},
            "numerical_validity": {"score": 0.0, "issues": [], "confidence": "Low"},
        },
        "primary_failure_mode": "unknown",
        "actionable_fix_list": [],
        "numeric_coverage_audit": {
            "missing_or_changed_numbers": [],
            "unchecked_numeric_groups": [],
            "coverage_confidence": "Low",
        },
        **{
            key: {
                "need_revise": 0,
                "uncertainty": 0,
                "error_type": "",
                "reason": "",
                "prompt": "",
            }
            for key in REVIEW_KEYS
        },
        "semantic_checks": {
            "tags": [],
            "issues": [],
            "suspicious_zero_solution": 0,
            "placeholder_or_missing_data": 0,
            "answer_metric_mismatch_risk": 0,
            "unit_scale_mismatch_risk": 0,
            "objective_metric_mismatch_risk": 0,
            "missing_business_conversion_factor": 0,
            "overall_semantic_risk": "low",
        },
    }


def dedupe_suggestions(items):
    """Normalize and de-duplicate textual suggestions while preserving their original order.
    
    Args:
        items: Sequence of values to normalize, summarize, or de-duplicate.
    
    Returns:
        The value described in the function summary.
    """
    seen = set()
    unique = []
    for item in items or []:
        text = str(item or "").strip()
        key = " ".join(text.split()).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(text)
    return unique


def format_suggestions_block(suggestions, title):
    """Format accumulated suggestions as a numbered prompt section.
    
    Args:
        suggestions: Accumulated natural-language repair suggestions.
        title: Input value used by this function.
    
    Returns:
        The value described in the function summary.
    """
    unique = dedupe_suggestions(suggestions)
    if not unique:
        return f"{title}\nNone."
    lines = [title]
    lines.extend(f"{idx}. {item}" for idx, item in enumerate(unique, 1))
    return "\n".join(lines)


def build_review_penalty_report(review):
    """Convert a component-wise LLM review into a numeric review score and penalty diagnostics.
    
    Args:
        review: Component-wise review dictionary returned by the LLM or rule checks.
    
    Returns:
        The value described in the function summary.
    """
    if not isinstance(review, dict):
        review = {}
    total_weight = sum(REVIEW_PENALTY_WEIGHTS.values())
    penalty = 0.0
    issues = []
    details = {}
    for key, weight in REVIEW_PENALTY_WEIGHTS.items():
        item = review.get(key, {})
        need_revise = isinstance(item, dict) and item.get("need_revise") == 1
        try:
            uncertainty = float(item.get("uncertainty", 0.5)) if isinstance(item, dict) else 0.5
        except (TypeError, ValueError):
            uncertainty = 0.5
        uncertainty = max(0.0, min(1.0, uncertainty))
        confidence = 1.0 - uncertainty
        layer_penalty = 0.0
        if need_revise:
            layer_penalty = weight * (0.5 + 0.5 * confidence)
            penalty += layer_penalty
            issues.append(key)
        details[key] = {
            "need_revise": 1 if need_revise else 0,
            "uncertainty": uncertainty,
            "weight": weight,
            "penalty": layer_penalty,
        }
    semantic_checks = review.get("semantic_checks", {}) if isinstance(review, dict) else {}
    semantic_penalty = 0.0
    semantic_issues = []
    if isinstance(semantic_checks, dict):
        hard_flags = (
            "placeholder_or_missing_data",
            "answer_metric_mismatch_risk",
            "unit_scale_mismatch_risk",
            "objective_metric_mismatch_risk",
            "missing_business_conversion_factor",
        )
        for key in hard_flags:
            if semantic_checks.get(key) in (1, True, "1", "true", "True"):
                semantic_issues.append(key)
        if semantic_checks.get("overall_semantic_risk") == "high":
            semantic_issues.append("overall_semantic_risk_high")
        semantic_penalty = min(0.35, 0.12 * len(set(semantic_issues)))
    numeric_audit = review.get("numeric_coverage_audit", {}) if isinstance(review, dict) else {}
    if isinstance(numeric_audit, dict):
        missing_numbers = numeric_audit.get("missing_or_changed_numbers")
        unchecked_groups = numeric_audit.get("unchecked_numeric_groups")
        if isinstance(missing_numbers, list) and missing_numbers:
            semantic_issues.append("numeric_coverage_mismatch")
            semantic_penalty = max(semantic_penalty, 0.30)
        if isinstance(unchecked_groups, list) and len(unchecked_groups) >= 2:
            semantic_issues.append("numeric_coverage_uncertain")
            semantic_penalty = max(semantic_penalty, 0.15)
    normalized_penalty = penalty / total_weight if total_weight else 0.0
    normalized_penalty = min(1.0, normalized_penalty + semantic_penalty)
    review_score = max(0.0, min(1.0, 1.0 - normalized_penalty))
    return {
        "review_score": review_score,
        "review_penalty": normalized_penalty,
        "review_issues": issues,
        "review_semantic_issues": sorted(set(semantic_issues)),
        "review_penalty_details": details,
    }


def apply_review_reward_caps(reward, review, score_report=None):
    """Conservatively cap no-GT reward when review/rule checks find hard risks."""
    try:
        capped = float(reward)
    except (TypeError, ValueError):
        capped = 0.0
    review = review if isinstance(review, dict) else {}
    score_report = score_report if isinstance(score_report, dict) else {}

    need_revise_layers = []
    for key in REVIEW_KEYS:
        layer_review = review.get(key)
        if isinstance(layer_review, dict) and layer_review.get("need_revise") == 1:
            need_revise_layers.append(key)
    if need_revise_layers:
        capped = min(capped, 0.82)
    if any(layer in need_revise_layers for layer in ("parameters", "objective", "constraints", "code")):
        capped = min(capped, 0.72)

    semantic_checks_value = review.get("semantic_checks")
    semantic_checks = semantic_checks_value if isinstance(semantic_checks_value, dict) else {}
    hard_review_flags = (
        "placeholder_or_missing_data",
        "answer_metric_mismatch_risk",
        "unit_scale_mismatch_risk",
        "objective_metric_mismatch_risk",
        "missing_business_conversion_factor",
    )
    if any(semantic_checks.get(key) == 1 for key in hard_review_flags):
        capped = min(capped, 0.68)
    if semantic_checks.get("overall_semantic_risk") == "high":
        capped = min(capped, 0.60)

    numeric_audit_value = review.get("numeric_coverage_audit")
    numeric_audit = numeric_audit_value if isinstance(numeric_audit_value, dict) else {}
    missing_or_changed_numbers = numeric_audit.get("missing_or_changed_numbers")
    unchecked_numeric_groups = numeric_audit.get("unchecked_numeric_groups")
    if isinstance(missing_or_changed_numbers, list) and missing_or_changed_numbers:
        capped = min(capped, 0.60)
    if isinstance(unchecked_numeric_groups, list) and len(unchecked_numeric_groups) >= 2:
        capped = min(capped, 0.78)

    semantic_report_value = score_report.get("semantic_report")
    semantic_report = semantic_report_value if isinstance(semantic_report_value, dict) else {}
    hard_rule_flags = (
        "placeholder_or_missing_data",
        "answer_metric_mismatch_risk",
        "unit_scale_mismatch_risk",
        "objective_metric_mismatch_risk",
        "missing_business_conversion_factor",
        "execution_semantic_failure",
    )
    if any(semantic_report.get(key) == 1 for key in hard_rule_flags):
        capped = min(capped, 0.70)
    if semantic_report.get("overall_semantic_risk") == "high":
        capped = min(capped, 0.60)

    return max(0.0, min(1.0, capped))


def _truncate_text(text, max_length=12000):
    """Shorten long text before placing it into prompts or diagnostic records.
    
    Args:
        text: Free-form text to inspect, truncate, or match.
        max_length: Input value used by this function.
    
    Returns:
        The value described in the function summary.
    """
    text = str(text or "")
    if len(text) <= max_length:
        return text
    head = max_length // 2
    tail = max_length - head
    return text[:head] + "\n... [truncated middle] ...\n" + text[-tail:]


def _as_list(value):
    """Normalize optional scalar or list-like values into a list.
    
    Args:
        value: Arbitrary value to normalize, inspect, parse, or serialize.
    
    Returns:
        The value described in the function summary.
    """
    if value is None:
        return []
    if isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], list):
            return value[0]
        return value
    return [value]


def _first_dict(value):
    """Return the first dictionary found in a value that may be a dictionary or list.
    
    Args:
        value: Arbitrary value to normalize, inspect, parse, or serialize.
    
    Returns:
        The value described in the function summary.
    """
    items = _as_list(value)
    for item in items:
        if isinstance(item, dict):
            return item
    return {}


def _is_constant_or_feasibility_objective(formulation):
    """Detect formulations whose objective is constant or only encodes feasibility.
    
    Args:
        formulation: Structured five-element optimization formulation.
    
    Returns:
        The value described in the function summary.
    """
    objective = _first_dict((formulation or {}).get("objective"))
    expression = str(objective.get("expression", "") or "").strip().lower()
    name = str(objective.get("name", "") or "").strip().lower()
    if not expression:
        return False

    compact = re.sub(r"\s+", "", expression)
    if compact in {"0", "0.0", "0.00"}:
        return True
    if re.fullmatch(r"[-+]?(?:\d+(?:\.0*)?|\.\d+)", compact):
        return True
    return "feasibility" in name or "feasibility" in expression


def _problem_has_optimizable_metric(problem_description):
    """Heuristically detect whether the problem asks for an optimizable metric.
    
    Args:
        problem_description: Natural-language optimization problem statement.
    
    Returns:
        The value described in the function summary.
    """
    text = str(problem_description or "").lower()
    metric_patterns = [
        r"\bminimi[sz]e\b",
        r"\bmaximi[sz]e\b",
        r"\bminimum\b",
        r"\bmaximum\b",
        r"\bprofit\b",
        r"\brevenue\b",
        r"\bcost\b",
        r"\bexpense",
        r"\bpollution\b",
        r"\bemission",
        r"\bwaste\b",
        r"\bmakespan\b",
        r"\bcycle\b",
        r"\btime\b",
        r"\btrip",
        r"\bvehicle",
        r"\bworker",
        r"\blabor\b",
        r"\braw material\b",
        r"\bmaterial use",
        r"\befficient",
        r"\befficiency\b",
    ]
    return any(re.search(pattern, text) for pattern in metric_patterns)


def objective_direction_from_formulation(formulation) -> str:
    """Infer the objective direction from a structured formulation.
    
    Args:
        formulation: Structured five-element optimization formulation.
    
    Returns:
        The value described in the function summary.
    """
    objective = _first_dict((formulation or {}).get("objective"))
    sense = str(objective.get("sense", "")).lower()
    return "maximize" if "max" in sense else "minimize"


def _is_finite_number(value) -> bool:
    """Check whether a value can be interpreted as a finite numeric value.
    
    Args:
        value: Arbitrary value to normalize, inspect, parse, or serialize.
    
    Returns:
        The value described in the function summary.
    """
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _normalize_objective_for_consensus(value: Any) -> Optional[float]:
    """Normalize an objective value into a finite float for consensus scoring.
    
    Args:
        value: Arbitrary value to normalize, inspect, parse, or serialize.
    
    Returns:
        The value described in the function summary.
    """
    if not _is_finite_number(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _cluster_objective_values(values, rel_tol=CONSENSUS_REL_TOL):
    """Group numerically similar objective values into tolerance-based clusters.
    
    Args:
        values: Sequence of values used for aggregation or matching.
        rel_tol: Relative tolerance for numeric comparison.
    
    Returns:
        The value described in the function summary.
    """
    normalized = sorted(v for v in (_normalize_objective_for_consensus(x) for x in values) if v is not None)
    clusters = []
    for value in normalized:
        placed = False
        for cluster in clusters:
            center = statistics.median(cluster)
            tolerance = rel_tol * max(1.0, abs(center), abs(value))
            if abs(value - center) <= tolerance:
                cluster.append(value)
                placed = True
                break
        if not placed:
            clusters.append([value])
    return clusters


def _mode_cluster_center(values, rel_tol=CONSENSUS_REL_TOL) -> Optional[float]:
    """Return the representative center of the largest objective-value cluster.
    
    Args:
        values: Sequence of values used for aggregation or matching.
        rel_tol: Relative tolerance for numeric comparison.
    
    Returns:
        The value described in the function summary.
    """
    clusters = _cluster_objective_values(values, rel_tol=rel_tol)
    if not clusters:
        return None
    ranked = sorted(clusters, key=lambda cluster: (len(cluster), -abs(statistics.median(cluster))), reverse=True)
    return float(statistics.median(ranked[0]))


def _relative_proximity_score(value: float, anchor: Optional[float], rel_tol=CONSENSUS_REL_TOL) -> float:
    """Score how close a value is to a consensus anchor under relative tolerance.
    
    Args:
        value: Arbitrary value to normalize, inspect, parse, or serialize.
        anchor: Input value used by this function.
        rel_tol: Relative tolerance for numeric comparison.
    
    Returns:
        The value described in the function summary.
    """
    normalized_value = _normalize_objective_for_consensus(value)
    normalized_anchor = _normalize_objective_for_consensus(anchor)
    if normalized_value is None or normalized_anchor is None:
        return 0.0
    denom = max(1.0, abs(normalized_anchor))
    rel_gap = abs(normalized_value - normalized_anchor) / denom
    if rel_gap <= rel_tol:
        return 1.0
    return max(0.0, math.exp(-6.0 * rel_gap))


def build_consensus_reward_report(
    objective_value,
    objective_history=None,
):
    """Build an auxiliary reward report based on agreement among historical objective values.
    
    Args:
        objective_value: Arbitrary value to normalize, inspect, parse, or serialize.
        objective_history: Historical objective values used for consensus scoring.
    
    Returns:
        The value described in the function summary.
    """
    history = [v for v in (objective_history or []) if _is_finite_number(v)]
    current_value = _normalize_objective_for_consensus(objective_value)
    report = {
        "enabled": False,
        "pool_size": len(history),
        "alpha": 0.0,
        "score_vote": 0.0,
        "score_stability": 0.0,
        "consensus_reward": 0.0,
        "mode_objective": None,
    }
    if current_value is None or len(history) < CONSENSUS_MIN_POOL:
        return report

    mode_value = _mode_cluster_center(history)
    score_vote = _relative_proximity_score(current_value, mode_value)
    clusters = _cluster_objective_values(history)
    matched_cluster_size = 0
    for cluster in clusters:
        center = statistics.median(cluster)
        tolerance = CONSENSUS_REL_TOL * max(1.0, abs(center), abs(current_value))
        if abs(current_value - center) <= tolerance:
            matched_cluster_size = max(matched_cluster_size, len(cluster))
    score_stability = matched_cluster_size / max(1, len(history))
    consensus_reward = 0.7 * score_vote + 0.3 * score_stability
    alpha = min(CONSENSUS_MAX_WEIGHT, CONSENSUS_MAX_WEIGHT * len(history) / max(1, CONSENSUS_WARMUP_POOL))
    report.update({
        "enabled": True,
        "pool_size": len(history),
        "alpha": alpha,
        "score_vote": score_vote,
        "score_stability": score_stability,
        "consensus_reward": consensus_reward,
        "mode_objective": mode_value,
    })
    return report


def formulation_to_5element_description(formulation) -> str:
    """Render a structured formulation as a human-readable five-element modeling description.
    
    Args:
        formulation: Structured five-element optimization formulation.
    
    Returns:
        The value described in the function summary.
    """
    sets = _as_list((formulation or {}).get("sets"))
    parameters = _as_list((formulation or {}).get("parameters"))
    variables = _as_list((formulation or {}).get("variables"))
    constraints = _as_list((formulation or {}).get("constraints"))
    objective = _first_dict((formulation or {}).get("objective"))

    lines = ["Sets:"]
    if sets:
        for item in sets:
            if isinstance(item, dict):
                lines.append(f"  {item.get('name', '<unnamed>')} = {item.get('elements', '<unspecified>')}")
            else:
                lines.append(f"  {item}")
    else:
        lines.append("  <missing>")

    lines.extend(["", "Variables:"])
    if variables:
        for item in variables:
            if isinstance(item, dict):
                suffix = f" indexed by {item['index_set']}" if item.get("index_set") else ""
                desc = f" // {item['description']}" if item.get("description") else ""
                lines.append(f"  {item.get('name', '<unnamed>')} in {item.get('domain', '<missing domain>')}{suffix}{desc}")
            else:
                lines.append(f"  {item}")
    else:
        lines.append("  <missing>")

    lines.extend(["", "Parameters:"])
    if parameters:
        for item in parameters:
            if isinstance(item, dict):
                if item.get("index_set"):
                    lines.append(f"  {item.get('name', '<unnamed>')} indexed by {item.get('index_set')}: {item.get('values', '<missing values>')}")
                else:
                    lines.append(f"  {item.get('name', '<unnamed>')} = {item.get('value', '<missing value>')}")
            else:
                lines.append(f"  {item}")
    else:
        lines.append("  <missing>")

    lines.extend(["", "Constraints:"])
    if constraints:
        for item in constraints:
            if isinstance(item, dict):
                desc = f" // {item['description']}" if item.get("description") else ""
                lines.append(f"  {item.get('name', '<unnamed>')}: {item.get('expression', item)}{desc}")
            else:
                lines.append(f"  {item}")
    else:
        lines.append("  <missing>")

    lines.extend(["", "Objective:"])
    if objective:
        lines.append(f"  {objective.get('sense', '<missing sense>').capitalize()} {objective.get('expression', objective)}")
    else:
        lines.append("  <missing>")

    return "\n".join(lines) + "\n"


def extract_objective_values_from_output(execution_output: str) -> list[float]:
    """Extract candidate objective values from solver or generated-code output text.
    
    Args:
        execution_output: Captured stdout/stderr from generated solver code.
    
    Returns:
        The value described in the function summary.
    """
    marker_match = re.search(r"__RERUN_OBJECTIVES__\s*(\[.*?\])", execution_output or "", flags=re.S)
    if marker_match:
        try:
            return [float(value) for value in json.loads(marker_match.group(1)) if isinstance(value, (int, float))]
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

    values = []
    number = r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    patterns = [
        rf"(?i)best\s+objective\s+({number})",
        rf"(?i)objective(?:\s+value)?(?:\s*\([^)]*\))?\s*[:=]\s*({number})",
        rf"(?i)optimal\s+objective(?:\s+value)?(?:\s*\([^)]*\))?\s*[:=]?\s*({number})",
        rf"(?im)^\s*(?:total|minimum|maximum|optimal)\s+(?:profit|cost|revenue|expense|expenses|value|fiber(?:\s+intake)?|fuel(?:\s+expenses)?|makespan)(?:\s*\([^)]*\))?\s*[:=]\s*({number})",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, execution_output or ""):
            try:
                values.append(float(str(match).replace(",", "")))
            except ValueError:
                continue
    return values


def extract_solver_status(execution_output: str, execution_error: str = "") -> str:
    """Extract a normalized solver status from execution output and errors.
    
    Args:
        execution_output: Captured stdout/stderr from generated solver code.
        execution_error: Error message produced while executing generated solver code.
    
    Returns:
        The value described in the function summary.
    """
    raw_text = f"{execution_output or ''}\n{execution_error or ''}"
    marker_match = re.search(
        r"__SOLVER_STATUS__\s*[:=]?\s*([A-Za-z_]+)",
        raw_text,
        flags=re.I,
    )
    if marker_match:
        status = marker_match.group(1).strip().lower()
        if status in {"optimal", "feasible", "time_limit", "infeasible", "unbounded", "unknown"}:
            return status
        if status in {"locallyoptimal", "locally_optimal", "globallyoptimal", "globally_optimal"}:
            return "optimal"
        if status in {"maxtimelimit", "max_time_limit", "timelimit"}:
            return "time_limit"

    text = raw_text.lower()
    if (
        "optimal solution found" in text
        or "termination_condition: optimal" in text
        or "termination condition: optimal" in text
        or "solver termination condition: optimal" in text
    ):
        return "optimal"
    if (
        "termination_condition: feasible" in text
        or "termination condition: feasible" in text
        or "solver termination condition: feasible" in text
    ):
        return "feasible"
    if "time limit" in text:
        return "time_limit"
    if "infeasible" in text:
        return "infeasible"
    if "unbounded" in text:
        return "unbounded"
    if "feasible solution" in text or "solution count" in text:
        return "feasible"
    return "unknown"


def _text_blob(*values):
    """Serialize arbitrary values into a lowercase text blob for semantic pattern checks.
    
    Args:
        values: Sequence of values used for aggregation or matching.
    
    Returns:
        The value described in the function summary.
    """
    parts = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            parts.append(value)
        else:
            try:
                parts.append(json.dumps(normalize_for_json(value), ensure_ascii=False))
            except Exception:
                parts.append(str(value))
    return "\n".join(parts)


def _has_any(text, patterns):
    """Return whether any regular-expression pattern matches a text blob.
    
    Args:
        text: Free-form text to inspect, truncate, or match.
        patterns: Regular-expression patterns used for text matching.
    
    Returns:
        The value described in the function summary.
    """
    text = str(text or "")
    return any(re.search(pattern, text, flags=re.I | re.S) for pattern in patterns)


def _count_numbers(text):
    """Count numeric literals present in text.
    
    Args:
        text: Free-form text to inspect, truncate, or match.
    
    Returns:
        The value described in the function summary.
    """
    return len(re.findall(r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", str(text or "")))


def _extract_answer_json(execution_output):
    """Extract the structured answer JSON marker emitted by generated solver code.
    
    Args:
        execution_output: Captured stdout/stderr from generated solver code.
    
    Returns:
        The value described in the function summary.
    """
    for line in str(execution_output or "").splitlines():
        if "__ANSWER_JSON__" not in line:
            continue
        payload = line.split("__ANSWER_JSON__", 1)[1].strip()
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None
    return None


def _objective_expression_text(formulation):
    """Collect objective expression text from a structured formulation.
    
    Args:
        formulation: Structured five-element optimization formulation.
    
    Returns:
        The value described in the function summary.
    """
    objective = _first_dict((formulation or {}).get("objective"))
    return str(objective.get("expression", "") or "")


def _business_conversion_risk(formulation, problem_l, model_l):
    """Conservative red flag for per-unit coefficients applied to aggregate-count variables."""
    objective_l = _objective_expression_text(formulation).lower()
    if not objective_l:
        return 0

    aggregate_count_terms = [
        r"\border", r"\borders", r"\bbatch", r"\bbatches", r"\bpackage", r"\bpackages",
        r"\btrip", r"\btrips", r"\bshipment", r"\bshipments", r"\bvehicle", r"\bvehicles",
        r"\btruck", r"\btrucks", r"\bbox", r"\bboxes", r"\bcontainer", r"\bcontainers",
    ]
    business_coeff_terms = [
        r"cost", r"price", r"profit", r"revenue", r"expense", r"wage",
        r"emission", r"pollution", r"capacity", r"material", r"labor",
    ]
    conversion_terms = [
        r"per[_ ]?order", r"per[_ ]?batch", r"per[_ ]?package", r"per[_ ]?trip",
        r"per[_ ]?shipment", r"per[_ ]?vehicle", r"per[_ ]?truck", r"per[_ ]?box",
        r"tables?per", r"items?per", r"units?per", r"products?per", r"kgper",
        r"capacity", r"load", r"batch[_ ]?size", r"quantity[_ ]?per",
    ]
    problem_conversion_phrases = [
        r"each\s+\w+\s+(?:order|batch|package|trip|shipment|vehicle|truck|box)\s+(?:includes?|contains?|holds?|carries?)",
        r"(?:per|for each)\s+(?:order|batch|package|trip|shipment|vehicle|truck|box)",
        r"\b\d+\s+(?:tables|items|units|products|kg|tons|people|passengers)\s+per\s+(?:order|batch|package|trip|shipment|vehicle|truck|box)",
    ]

    has_aggregate_variable = _has_any(objective_l, aggregate_count_terms)
    has_business_coeff = _has_any(objective_l, business_coeff_terms)
    conversion_available = _has_any(model_l, conversion_terms) or _has_any(problem_l, problem_conversion_phrases)
    conversion_used_in_objective = _has_any(objective_l, conversion_terms)

    if has_aggregate_variable and has_business_coeff and conversion_available and not conversion_used_in_objective:
        return 1
    return 0


def _decision_vector_requested(problem_l):
    """Detect whether the problem asks for a full decision vector rather than only an objective value.
    
    Args:
        problem_l: Input value used by this function.
    
    Returns:
        The value described in the function summary.
    """
    return _has_any(problem_l, [
        r"\bhow many\b.{0,80}\b(?:each|of each|and)\b",
        r"\bhow much\b.{0,80}\b(?:each|of each|and)\b",
        r"\bhow much money\b.{0,80}\beach\b",
        r"\bdetermine\b.{0,80}\b(?:assignment|allocation|schedule|plan|mix)\b",
        r"\bwhat\b.{0,80}\b(?:amounts|quantities|levels)\b",
    ])


def _answer_variables_count(answer_json):
    """Count decision-variable entries reported in the structured answer JSON.
    
    Args:
        answer_json: Input value used by this function.
    
    Returns:
        The value described in the function summary.
    """
    variables = answer_json.get("variables") if isinstance(answer_json, dict) else None
    if isinstance(variables, dict):
        return len([key for key, value in variables.items() if str(key).strip() and value is not None])
    if isinstance(variables, list):
        return len([item for item in variables if item is not None])
    return 0


def _count_like_relaxation_risk(formulation, problem_l, model_l):
    """Detect possible continuous relaxation of count-like integer decisions.
    
    Args:
        formulation: Structured five-element optimization formulation.
        problem_l: Input value used by this function.
        model_l: Input value used by this function.
    
    Returns:
        The value described in the function summary.
    """
    count_terms = [
        r"\bmachine(?:s)?\b", r"\bvehicle(?:s)?\b", r"\btruck(?:s)?\b", r"\btrip(?:s)?\b",
        r"\bbag(?:s)?\b", r"\bmeal(?:s)?\b", r"\bbatch(?:es)?\b", r"\bworker(?:s)?\b",
        r"\bshift(?:s)?\b", r"\bcontainer(?:s)?\b", r"\bpill(?:s)?\b", r"\bgumm(?:y|ies)\b",
        r"\bcatalyst(?:s)?\b", r"\bpackage(?:s)?\b", r"\border(?:s)?\b",
        r"\bitem(?:s)?\b", r"\bselected\b", r"\bvisit(?:ed|s)?\b",
    ]
    divisible_terms = [
        r"\bfraction(?:al)?\b", r"\bcontinuous\b", r"\bdivisible\b", r"\bhour(?:s)?\b",
        r"\btime\b", r"\bmoney\b", r"\binvest(?:ed|ment)?\b", r"\bflow\b",
        r"\bweight\b", r"\bvolume\b", r"\bkg\b", r"\bton(?:s)?\b", r"\bpercent(?:age)?\b",
        r"\bproportion\b", r"\barea\b", r"\bpower\b", r"\bmaterial\b",
    ]
    explicit_fractional_permission = _has_any(problem_l, [r"\bfraction(?:al)?\b", r"\bcontinuous\b", r"\bdivisible\b"])
    if not _has_any(problem_l, count_terms) or explicit_fractional_permission:
        return 0

    variables = _as_list((formulation or {}).get("variables"))
    for variable in variables:
        if not isinstance(variable, dict):
            continue
        domain = str(variable.get("domain", "") or "").lower()
        if domain not in {"reals", "nonnegativereals"}:
            continue
        var_text = _text_blob(variable).lower()
        if _has_any(var_text, divisible_terms):
            continue
        if _has_any(var_text, count_terms) or _has_any(problem_l, [r"\bhow many\b"]):
            return 1

    if _has_any(model_l, [r"domain\s*=\s*NonNegativeReals", r"within\s*=\s*NonNegativeReals"]) and _has_any(problem_l, [r"\bhow many\b"]):
        return 1
    return 0


def _profit_double_subtraction_risk(formulation, problem_l):
    """Detect common profit-modeling mistakes involving double subtraction of costs.
    
    Args:
        formulation: Structured five-element optimization formulation.
        problem_l: Input value used by this function.
    
    Returns:
        The value described in the function summary.
    """
    objective_l = _objective_expression_text(formulation).lower()
    if not objective_l:
        return 0
    profit_wording = _has_any(problem_l, [
        r"\bsold\b.{0,60}\bprofit\b",
        r"\bprofit\s+of\b",
        r"\bprofits?\b.{0,30}\bper\b",
        r"\bearns?\b.{0,40}\bprofit\b",
        r"\bnet revenue\b",
        r"\bnet profit per\b",
    ])
    subtracts_cost = "-" in objective_l and _has_any(objective_l, [
        r"\bcost\b", r"\bpurchase(?:_|\s)?cost\b", r"\bpurchase\b",
        r"\bacquisition(?:_|\s)?cost\b", r"\bacquisition\b", r"\bbuy\b", r"\braw(?:_|\s)?material\b",
    ])
    return 1 if profit_wording and subtracts_cost else 0


def _tsp_connectivity_risk(problem_l, model_l):
    """Detect route or TSP formulations that may lack connectivity or subtour-prevention logic.
    
    Args:
        problem_l: Input value used by this function.
        model_l: Input value used by this function.
    
    Returns:
        The value described in the function summary.
    """
    tsp_like = _has_any(problem_l, [
        r"\btravel(?:ing|ling)? salesman\b",
        r"\bTSP\b",
        r"\btour\b",
        r"\bvisit\b.{0,80}\beach\b.{0,40}\bcit(?:y|ies)\b",
        r"\beach\b.{0,40}\bcit(?:y|ies)\b.{0,80}\bexactly once\b",
    ])
    if not tsp_like:
        return 0
    has_degree = _has_any(model_l, [r"\bincoming\b", r"\boutgoing\b", r"\bdegree\b", r"==\s*1"])
    has_connectivity = _has_any(model_l, [
        r"\bsubtour\b", r"\bMTZ\b", r"\bmiller", r"\border\b", r"\brank\b",
        r"\bconnect", r"\bflow\b", r"\bcutset\b", r"\bu\s*\[",
    ])
    return 1 if has_degree and not has_connectivity else 0


def build_semantic_report(formulation, problem_description="", pyomo_code="", execution_output="", objective_value=None):
    """Ground-truth-free semantic checks for common OR modeling failure modes."""
    problem = str(problem_description or "")
    model_text = _text_blob(formulation, pyomo_code)
    output = str(execution_output or "")
    problem_l = problem.lower()
    model_l = model_text.lower()
    output_l = output.lower()

    tags = []
    issues = []

    tag_patterns = {
        "goal_programming": [r"\bgoal(?: programming|s)?\b", r"\bpriority\b", r"\bdeviation\b", r"\bp[_\{\s]*\d", r"\btarget\b.{0,80}\bdeviation\b", r"\btarget\b.{0,80}\bpriority\b"],
        "multi_period": [r"\bperiod\b", r"\bmonth\b", r"\bquarter\b", r"\byear\b", r"\bweek\b", r"\binventory\b", r"\bstorage\b", r"\bbacklog\b", r"\bdelayed?\b", r"\bcarry[- ]?over\b"],
        "workforce": [r"\bworker\b", r"\bworkforce\b", r"\bhir(?:e|ing)\b", r"\btrain(?:ing|ee)?\b", r"\bskilled\b", r"\blabor force\b"],
        "blending": [r"\bblend", r"\bmixture\b", r"\bcrude\b", r"\bgasoline\b", r"\bsulfur\b", r"\bingredient\b", r"\bcomposition\b", r"\bcontent\b", r"\bpercentage\b"],
        "cutting_stock": [r"\bcutting pattern\b", r"\broll\b", r"\bpipe\b", r"\bstock length\b", r"\bleftover\b", r"\btrim loss\b"],
        "routing_network": [r"\broute\b", r"\bvehicle\b", r"\bdepot\b", r"\btime window\b", r"\bTSP\b", r"\bVRP\b", r"\bnode\b", r"\barc\b", r"\bedge\b", r"\bbandwidth\b", r"\bsource\b", r"\bsink\b"],
        "assignment": [r"\bassign(?:ment|ed)?\b", r"\bone[- ]to[- ]one\b", r"\bmatching\b"],
        "project_scheduling": [r"\bactivity\b", r"\bduration\b", r"\bpredecessor\b", r"\bcritical path\b", r"\bproject\b"],
        "job_shop_scheduling": [r"\bjob[- ]?shop\b", r"\boperation\b", r"\bsame machine\b", r"\bprocessing time\b", r"\bmakespan\b", r"\bprecedence\b"],
        "aircraft_landing": [r"\baircraft\b", r"\bplane\b", r"\blanding\b", r"\brunway\b", r"\bearliest\b", r"\blatest\b", r"\bseparation\b"],
    }
    for tag, patterns in tag_patterns.items():
        if _has_any(problem_l, patterns):
            tags.append(tag)

    if "multi_period" in tags:
        if not _has_any(model_l, [r"\bperiod", r"\bmonth", r"\bquarter", r"\byear", r"\bweek", r"\btime"]):
            issues.append("multi_period_problem_without_time_index")
        if _has_any(problem_l, [r"\binventory\b", r"\bstorage\b", r"\bstock\b"]) and not _has_any(model_l, [r"\binventory\b", r"\bstorage\b", r"\bstock\b"]):
            issues.append("inventory_or_storage_mentioned_without_state_variable")
        if _has_any(problem_l, [r"\binitial\b", r"\bbeginning\b", r"\bopening\b"]) and not _has_any(model_l, [r"\binitial\b", r"\bbeginning\b", r"\bopening\b", r"\bt\s*==\s*0", r"\bfirst\b"]):
            issues.append("initial_state_condition_not_evident")
        if _has_any(problem_l, [r"\bending\b", r"\bfinal\b", r"\bend of\b", r"\bat the end\b"]) and not _has_any(model_l, [r"\bending\b", r"\bfinal\b", r"\bend\b", r"\blast\b"]):
            issues.append("final_state_condition_not_evident")
        if _has_any(problem_l, [r"\bbacklog\b", r"\bdelay(?:ed)?\b"]) and not _has_any(model_l, [r"\bbacklog\b", r"\bdelay", r"\blate\b"]):
            issues.append("backlog_or_delay_state_not_evident")
        needs_transition = _has_any(problem_l, [r"\binventory\b", r"\bstorage\b", r"\bstock\b", r"\bbacklog\b", r"\bcarry[- ]?over\b", r"\bstate\b"])
        has_transition = _has_any(model_l, [
            r"\bt\s*-\s*1\b",
            r"\bperiod\s*-\s*1\b",
            r"\bprevious\b",
            r"\bprev\b",
            r"\bcarry[- ]?over\b",
            r"\bbalance\b",
            r"\bflow conservation\b",
            r"\bstate transition\b",
        ])
        if needs_transition and not has_transition:
            issues.append("multi_period_state_transition_not_evident")

    if "goal_programming" in tags:
        has_deviation = _has_any(model_l, [r"\bdeviation\b", r"\bd_plus\b", r"\bd_minus\b", r"\boverachievement\b", r"\bunderachievement\b"])
        has_priority = _has_any(model_l, [r"\bpriority\b", r"\bweight", r"\blexicographic\b", r"\bgoal\b"])
        if not has_deviation:
            issues.append("goal_programming_without_deviation_variables")
        if not has_priority:
            issues.append("goal_programming_without_priority_or_weight_semantics")
        if not _has_any(model_l, [r"\brequested metric\b", r"\banswer_metric_name\b", r"\bderived_metrics\b", r"\bbusiness metric\b", r"\breported metric\b"]):
            issues.append("goal_programming_requested_metric_not_distinguished_from_internal_objective")

    if "blending" in tags:
        if not _has_any(model_l, [r"\bbalance\b", r"\bflow\b", r"\bamount\b", r"\bquantity\b"]):
            issues.append("blending_without_material_balance")
        if _has_any(problem_l, [r"\bsulfur\b", r"\bcontent\b", r"\bquality\b", r"\bpercentage\b", r"%"]) and not _has_any(model_l, [r"\bsulfur\b", r"\bcontent\b", r"\bquality\b", r"\bpercentage\b", r"\bcomposition\b"]):
            issues.append("blending_quality_constraints_not_evident")

    if "cutting_stock" in tags:
        if re.search(r"\bVar\s*\([^)]*\)\s*\*\s*\bVar\b|\bmodel\.\w+\[[^\]]+\]\s*\*\s*model\.\w+\[[^\]]+\]", model_text):
            issues.append("possible_unlinearized_variable_product")
        if not _has_any(model_l, [r"\bdemand\b", r"\border\b", r"\brequire"]):
            issues.append("cutting_stock_without_demand_satisfaction")
        if not _has_any(model_l, [r"\blength\b", r"\bwidth\b", r"\bcapacity\b"]):
            issues.append("cutting_stock_without_stock_length_constraint")
        if _has_any(problem_l, [r"\bleftover\b", r"\btrim\b", r"\bwaste\b"]) and not _has_any(model_l, [r"\bleftover\b", r"\btrim\b", r"\bwaste\b"]):
            issues.append("leftover_or_trim_limit_not_evident")

    if "assignment" in tags and _has_any(problem_l, [r"\bone[- ]to[- ]one\b", r"exactly one", r"each .* exactly one", r"\beach\b"]):
        exactly_one_count = len(re.findall(r"==\s*1|equals?\s+1|exactly\s+one", model_l))
        has_two_sides = _has_any(model_l, [r"\bfor each\b", r"\beach\b"]) and exactly_one_count >= 2
        if not has_two_sides:
            issues.append("one_to_one_assignment_missing_both_exactly_one_sides")
        if not _has_any(model_l, [r"\bbinary\b", r"\bwithin=Binary\b", r"\bdomain=Binary\b", r"\b0-1\b"]):
            issues.append("assignment_without_binary_decision_variable")

    if "job_shop_scheduling" in tags or ("project_scheduling" in tags and _has_any(problem_l, [r"\bmachine\b", r"\boperation\b", r"\bmakespan\b"])):
        tags.append("job_shop_scheduling")
        if not _has_any(model_l, [r"\bstart\b", r"\bstart_time\b", r"\bstarttime\b"]):
            issues.append("scheduling_without_start_time_variables")
        if _has_any(problem_l, [r"\bprecedence\b", r"\bpredecessor\b", r"\bbefore\b"]) and not _has_any(model_l, [r"\bprecedence\b", r"\bpredecessor\b", r"\bbefore\b", r"\bfinish\b"]):
            issues.append("scheduling_precedence_constraints_not_evident")
        if _has_any(problem_l, [r"\bmachine\b", r"\bsame machine\b", r"\boperation\b"]) and not _has_any(model_l, [r"\bno[-_ ]?overlap\b", r"\bdisjunct", r"\bsequence\b", r"\border\b", r"\bbig[-_ ]?m\b"]):
            issues.append("scheduling_machine_no_overlap_not_evident")
        if _has_any(problem_l, [r"\bmachine\b", r"\bsame machine\b"]) and not _has_any(model_l, [r"\bpair\b", r"\bconflict\b", r"\bsequence\[[^\]]*,[^\]]*\]", r"\border\[[^\]]*,[^\]]*\]"]):
            issues.append("scheduling_pairwise_order_binary_not_evident")

    if "aircraft_landing" in tags:
        if _has_any(problem_l, [r"\bearliest\b"]) and not _has_any(model_l, [r"\bearliest\b", r"\brelease\b", r"\blower bound\b"]):
            issues.append("aircraft_landing_earliest_time_not_evident")
        if _has_any(problem_l, [r"\blatest\b"]) and not _has_any(model_l, [r"\blatest\b", r"\bdue\b", r"\bupper bound\b"]):
            issues.append("aircraft_landing_latest_time_not_evident")
        if _has_any(problem_l, [r"\btarget\b"]) and not _has_any(model_l, [r"\btarget\b", r"\bdesired\b"]):
            issues.append("aircraft_landing_target_time_not_evident")
        if _has_any(problem_l, [r"\bearly\b", r"\blate\b", r"\bpenalt"]) and not _has_any(model_l, [r"\btarget\b.*\bpenalt", r"\bpenalt.*\btarget\b", r"\bearly\b", r"\blate\b"]):
            issues.append("aircraft_landing_penalty_may_not_be_target_based")
        if _has_any(problem_l, [r"\bseparation\b", r"\brunway\b"]) and not _has_any(model_l, [r"\bseparation\b", r"\bsequence\b", r"\border\b", r"\bbig[-_ ]?m\b", r"\bpair\b"]):
            issues.append("aircraft_landing_pairwise_separation_not_evident")

    if "routing_network" in tags:
        if _has_any(problem_l, [r"\btime window\b"]) and not _has_any(model_l, [r"\btime\b", r"\bearliest\b", r"\blatest\b", r"\bwindow\b"]):
            issues.append("time_window_problem_without_time_variables_or_bounds")
        if _has_any(problem_l, [r"\broute\b", r"\bvehicle\b", r"\bTSP\b", r"\bVRP\b"]) and not _has_any(model_l, [r"\bflow\b", r"\bvisit\b", r"\bsubtour\b", r"\bconnect", r"\bdegree\b"]):
            issues.append("routing_problem_without_connectivity_or_visit_structure")
    if _tsp_connectivity_risk(problem_l, model_l):
        issues.append("tsp_degree_constraints_without_subtour_elimination")

    if _has_any(problem_l, [r"\btier", r"\bstep", r"\bdiscount", r"\bpiecewise", r"\bblock price", r"\bincreasing price", r"\bdecreasing price"]):
        tags.append("piecewise_tiered")
        if not _has_any(model_l, [r"\bbinary\b", r"\btier", r"\bstep", r"\bpiecewise", r"\bbig[-_ ]?m", r"\bblock"]):
            issues.append("piecewise_or_tiered_semantics_not_evident")

    unit_scale_mismatch_risk = _business_conversion_risk(formulation, problem_l, model_l)
    missing_business_conversion_factor = unit_scale_mismatch_risk
    if unit_scale_mismatch_risk:
        issues.append("possible_missing_business_conversion_factor_in_objective")

    objective_metric_mismatch_risk = 0
    objective_l = _objective_expression_text(formulation).lower()
    if _has_any(problem_l, [r"\bnet profit\b", r"\btotal net profit\b"]) and not _has_any(objective_l, [r"\bcost\b", r"\bexpense\b", r"\bwage\b", r"\binventory\b", r"\bholding\b", r"\bstockout\b", r"\boutsource\b", r"\bhir", r"\bfire"]):
        objective_metric_mismatch_risk = 1
        issues.append("net_profit_objective_missing_cost_components")
    if _has_any(problem_l, [r"\bfixed cost", r"\bfixed costs"]) and _has_any(problem_l, [r"\bprofit\b", r"\bcost\b"]) and not _has_any(objective_l, [r"\bfixed\b", r"\bactivate\b", r"\buse\b", r"\bbinary\b"]):
        objective_metric_mismatch_risk = 1
        issues.append("fixed_costs_not_evident_in_objective")
    if _profit_double_subtraction_risk(formulation, problem_l):
        objective_metric_mismatch_risk = 1
        issues.append("profit_or_net_revenue_may_be_double_subtracted")

    integer_domain_relaxation_risk = _count_like_relaxation_risk(formulation, problem_l, model_l)
    if integer_domain_relaxation_risk:
        issues.append("count_like_decision_relaxed_to_continuous")

    placeholder_or_missing_data = 0
    if _has_any(model_l, [r"placeholder (?:data|value|coefficient|parameter|cost|price)", r"\bTODO\b", r"replace with actual", r"not provided", r"actual coefficients"]):
        placeholder_or_missing_data = 1
        issues.append("placeholder_or_dummy_data_in_model")
    if _has_any(problem_l, [r"table below", r"shown in table", r"as follows"]) and _count_numbers(problem) < 4 and _has_any(model_l, [r"\b0(?:\.0)?\b", r"\bplaceholder\b", r"not provided"]):
        placeholder_or_missing_data = 1
        issues.append("table_referenced_but_numeric_data_not_evident")

    suspicious_zero_solution = 0
    if objective_value is None:
        objective_values = extract_objective_values_from_output(output)
        objective_value = objective_values[0] if objective_values else None
    try:
        objective_float = float(objective_value) if objective_value is not None else None
    except (TypeError, ValueError):
        objective_float = None
    asks_positive_work = _has_any(problem_l, [
        r"\bmaximize\b", r"\bminimize\b", r"\bprofit\b", r"\brevenue\b", r"\bproduction\b",
        r"\bexport\b", r"\bdemand\b", r"\brequired\b", r"\bat least\b", r"\btransport\b",
        r"\bdeliver\b", r"\bsupply\b", r"\bcost\b", r"\bmake\b", r"\bproduce\b",
    ])
    if (
        "goal_programming" not in tags
        and objective_float is not None
        and math.isfinite(objective_float)
        and abs(objective_float) <= 1e-9
        and asks_positive_work
    ):
        if _has_any(output_l, [r"objective 0", r"best objective\s+0", r'"objective"\s*:\s*-?0(?:\.0)?', r'"answer_value"\s*:\s*-?0(?:\.0)?']):
            suspicious_zero_solution = 1
            issues.append("suspicious_zero_objective_for_optimizing_problem")

    solver_status = extract_solver_status(output, "")
    execution_semantic_failure = 0
    if solver_status in {"infeasible", "unbounded"}:
        execution_semantic_failure = 1
        issues.append(f"solver_status_{solver_status}")
    if solver_status in {"optimal", "feasible"} and objective_float is None:
        execution_semantic_failure = 1
        issues.append("missing_objective_value_after_feasible_solve")

    answer_metric_mismatch_risk = 0
    answer_json = _extract_answer_json(output)
    decision_vector_requested = _decision_vector_requested(problem_l)
    if isinstance(answer_json, dict):
        answer_type = str(answer_json.get("answer_type", "") or "").lower()
        derived_metrics = answer_json.get("derived_metrics")
        answer_metric_name = str(answer_json.get("answer_metric_name", "") or "").strip()
        why_matches = str(answer_json.get("why_this_matches_question", "") or "").strip()
        if "goal_programming" in tags and answer_type == "objective" and not derived_metrics:
            answer_metric_mismatch_risk = 1
            issues.append("goal_programming_answer_metric_not_explained")
        if "goal_programming" in tags and (not answer_metric_name or not why_matches):
            answer_metric_mismatch_risk = 1
            issues.append("goal_programming_answer_metric_fields_missing")
        if "goal_programming" in tags and answer_type == "objective" and _has_any(str(answer_metric_name).lower(), [r"\bobjective\b", r"\bweighted\b", r"\bdeviation\b"]):
            answer_metric_mismatch_risk = 1
            issues.append("goal_programming_answer_value_appears_to_be_internal_objective")
        if _has_any(problem_l, [r"\bhow many\b", r"\bhow much of each\b", r"\bdetermine the assignment\b"]) and not answer_json.get("variables"):
            answer_metric_mismatch_risk = 1
            issues.append("question_requests_decisions_but_answer_json_lacks_variables")
        if decision_vector_requested and answer_type == "objective":
            answer_metric_mismatch_risk = 1
            issues.append("decision_vector_question_reported_as_objective")
        if decision_vector_requested and _answer_variables_count(answer_json) <= 1:
            answer_metric_mismatch_risk = 1
            issues.append("decision_vector_answer_missing_complete_variable_plan")
        if not answer_metric_name or not why_matches:
            answer_metric_mismatch_risk = 1
            issues.append("answer_metric_identity_not_documented")
    elif "__ANSWER_JSON__" in output:
        answer_metric_mismatch_risk = 1
        issues.append("malformed_answer_json")

    issue_count = len(set(issues))
    score = 1.0
    score -= min(0.45, 0.10 * issue_count)
    if placeholder_or_missing_data:
        score -= 0.25
    if suspicious_zero_solution:
        score -= 0.20
    if answer_metric_mismatch_risk:
        score -= 0.25
    if unit_scale_mismatch_risk:
        score -= 0.20
    if objective_metric_mismatch_risk:
        score -= 0.20
    if integer_domain_relaxation_risk:
        score -= 0.15
    if execution_semantic_failure:
        score -= 0.30
    score = max(0.0, min(1.0, score))
    high_issue_codes = {
        "goal_programming_answer_value_appears_to_be_internal_objective",
        "goal_programming_answer_metric_not_explained",
        "aircraft_landing_penalty_may_not_be_target_based",
        "aircraft_landing_pairwise_separation_not_evident",
        "scheduling_pairwise_order_binary_not_evident",
        "multi_period_state_transition_not_evident",
        "possible_missing_business_conversion_factor_in_objective",
        "net_profit_objective_missing_cost_components",
        "fixed_costs_not_evident_in_objective",
        "count_like_decision_relaxed_to_continuous",
        "decision_vector_question_reported_as_objective",
        "decision_vector_answer_missing_complete_variable_plan",
        "profit_or_net_revenue_may_be_double_subtracted",
        "tsp_degree_constraints_without_subtour_elimination",
        "solver_status_infeasible",
        "solver_status_unbounded",
        "missing_objective_value_after_feasible_solve",
    }
    if placeholder_or_missing_data or suspicious_zero_solution or execution_semantic_failure or issue_count >= 4 or any(issue in high_issue_codes for issue in issues):
        risk = "high"
    elif answer_metric_mismatch_risk or unit_scale_mismatch_risk or objective_metric_mismatch_risk or issue_count >= 2:
        risk = "medium"
    else:
        risk = "low"

    return {
        "score": score,
        "tags": sorted(set(tags)),
        "issues": sorted(set(issues)),
        "suspicious_zero_solution": suspicious_zero_solution,
        "placeholder_or_missing_data": placeholder_or_missing_data,
        "answer_metric_mismatch_risk": answer_metric_mismatch_risk,
        "unit_scale_mismatch_risk": unit_scale_mismatch_risk,
        "objective_metric_mismatch_risk": objective_metric_mismatch_risk,
        "integer_domain_relaxation_risk": integer_domain_relaxation_risk,
        "missing_business_conversion_factor": missing_business_conversion_factor,
        "execution_semantic_failure": execution_semantic_failure,
        "overall_semantic_risk": risk,
    }


def _extract_gap_percent(execution_output: str):
    """Extract an optimization gap percentage from solver logs.
    
    Args:
        execution_output: Captured stdout/stderr from generated solver code.
    
    Returns:
        The value described in the function summary.
    """
    match = re.search(r"gap\s+([0-9.]+)%", execution_output or "", flags=re.I)
    return float(match.group(1)) if match else None


def _extract_nodes_explored(execution_output: str):
    """Extract the number of explored branch-and-bound nodes from solver logs.
    
    Args:
        execution_output: Captured stdout/stderr from generated solver code.
    
    Returns:
        The value described in the function summary.
    """
    match = re.search(r"Explored\s+(\d+)\s+nodes", execution_output or "", flags=re.I)
    return int(match.group(1)) if match else None


def _definition_score(formulation) -> tuple[float, list[str]]:
    """Score whether a formulation contains the required five modeling elements.
    
    Args:
        formulation: Structured five-element optimization formulation.
    
    Returns:
        The value described in the function summary.
    """
    issues = []
    required = ("sets", "parameters", "variables", "objective", "constraints")
    present = 0
    for key in required:
        value = (formulation or {}).get(key)
        ok = bool(value)
        if key == "objective":
            objective = _first_dict(value)
            ok = bool(objective.get("sense") and objective.get("expression"))
        if ok:
            present += 1
        else:
            issues.append(f"missing_or_incomplete_{key}")
    return present / len(required), issues


def _structure_score(formulation) -> tuple[float, list[str]]:
    """Score structural quality and internal consistency of a formulation.
    
    Args:
        formulation: Structured five-element optimization formulation.
    
    Returns:
        The value described in the function summary.
    """
    checks = []
    issues = []

    for idx, item in enumerate(_as_list((formulation or {}).get("sets"))):
        ok = isinstance(item, dict) and bool(item.get("name")) and "elements" in item
        checks.append(ok)
        if not ok:
            issues.append(f"set_{idx}_missing_name_or_elements")

    for idx, item in enumerate(_as_list((formulation or {}).get("parameters"))):
        ok = isinstance(item, dict) and bool(item.get("name")) and ("value" in item or "values" in item)
        checks.append(ok)
        if not ok:
            issues.append(f"parameter_{idx}_missing_name_or_value")

    for idx, item in enumerate(_as_list((formulation or {}).get("variables"))):
        ok = isinstance(item, dict) and bool(item.get("name")) and bool(item.get("domain"))
        checks.append(ok)
        if not ok:
            issues.append(f"variable_{idx}_missing_name_or_domain")

    for idx, item in enumerate(_as_list((formulation or {}).get("constraints"))):
        ok = isinstance(item, dict) and bool(item.get("name")) and bool(item.get("expression"))
        checks.append(ok)
        if not ok:
            issues.append(f"constraint_{idx}_missing_name_or_expression")

    objective = _first_dict((formulation or {}).get("objective"))
    checks.append(bool(objective.get("sense") and objective.get("expression")))
    if not checks[-1]:
        issues.append("objective_missing_sense_or_expression")

    if not checks:
        return 0.0, ["empty_structure"]
    return sum(1 for item in checks if item) / len(checks), issues


def build_selection_score_report(formulation, execution_output="", execution_error="", source="mcts_final", problem_description="", pyomo_code=""):
    """Combine execution, solver, semantic, and structural evidence into a candidate-selection score.
    
    Args:
        formulation: Structured five-element optimization formulation.
        execution_output: Captured stdout/stderr from generated solver code.
        execution_error: Error message produced while executing generated solver code.
        source: Input value used by this function.
        problem_description: Natural-language optimization problem statement.
        pyomo_code: Generated Python/Pyomo source code to execute or review.
    
    Returns:
        The value described in the function summary.
    """
    status = extract_solver_status(execution_output, execution_error)
    objective_values = extract_objective_values_from_output(execution_output)
    objective_value = objective_values[0] if objective_values else None
    def_score, def_issues = _definition_score(formulation)
    str_score, str_issues = _structure_score(formulation)
    exec_ok = not execution_error and "traceback (most recent call last)" not in (execution_output or "").lower()
    semantic_report = build_semantic_report(
        formulation,
        problem_description=problem_description,
        pyomo_code=pyomo_code,
        execution_output=execution_output,
        objective_value=objective_value,
    )

    solve_scores = {
        "optimal": 1.0,
        "feasible": 0.75,
        "time_limit": 0.5,
        "unknown": 0.2 if exec_ok else 0.0,
        "infeasible": 0.0,
        "unbounded": 0.0,
    }
    solve_score = solve_scores.get(status, 0.0)
    obj_score = 1.0 if objective_value is not None else (0.5 if status in {"optimal", "feasible"} else 0.0)
    objective_issues = []
    if _is_constant_or_feasibility_objective(formulation) and _problem_has_optimizable_metric(problem_description):
        obj_score = min(obj_score, 0.2)
        objective_issues.append("constant_or_feasibility_objective_with_optimizable_metric")
    exec_score = 1.0 if exec_ok else 0.0

    score_components = {
        "exec": exec_score,
        "solve": solve_score,
        "obj": obj_score,
        "model": (def_score + str_score) / 2.0,
        "def": def_score,
        "str": str_score,
        "num": solve_score,
    }
    overall_score = sum(score_components[key] * SELECTION_WEIGHTS[key] for key in SCORE_KEYS)
    semantic_score = semantic_report.get("score", 1.0)
    if semantic_report.get("overall_semantic_risk") == "high":
        overall_score = min(overall_score, 0.55)
    elif semantic_score < 0.5:
        overall_score = min(overall_score, 0.55)
    elif semantic_score < 0.75:
        overall_score = min(overall_score, 0.8)
    if semantic_report.get("answer_metric_mismatch_risk") == 1:
        overall_score = min(overall_score, 0.60)
    if (
        semantic_report.get("unit_scale_mismatch_risk") == 1
        or semantic_report.get("objective_metric_mismatch_risk") == 1
        or semantic_report.get("missing_business_conversion_factor") == 1
    ):
        overall_score = min(overall_score, 0.75)
    if semantic_report.get("execution_semantic_failure") == 1:
        overall_score = min(overall_score, 0.35)
    if status in {"infeasible", "unbounded"}:
        overall_score = min(overall_score, 0.25)
    if status == "unknown":
        overall_score = min(overall_score, 0.60)
    if objective_value is None and status in {"optimal", "feasible"}:
        overall_score = min(overall_score, 0.50)

    if not exec_ok:
        primary_failure_mode = "execution_error"
    elif status in {"infeasible", "unbounded"}:
        primary_failure_mode = status
    elif objective_value is None:
        primary_failure_mode = "missing_objective_value"
    else:
        primary_failure_mode = "unknown"

    evidence = {
        "source": source,
        "objective_direction": objective_direction_from_formulation(formulation),
        "optimal_value": objective_value,
        "ground_truth_used": False,
        "solve_status": status,
        "gap_percent": _extract_gap_percent(execution_output),
        "nodes_explored": _extract_nodes_explored(execution_output),
        "best_objective": objective_value,
        "model_components": {
            "rule_score": (def_score + str_score) / 2.0,
            "overall_score": (def_score + str_score) / 2.0,
            "confidence": "High" if def_score >= 0.9 and str_score >= 0.9 else "Medium",
        },
        "rel_components": {
            "has_both_sources": False,
        },
        "error_message": execution_error,
    }

    return {
        "overall_score": overall_score,
        "score_components": score_components,
        "weights": dict(SELECTION_WEIGHTS),
        "evidence": evidence,
        "primary_failure_mode": primary_failure_mode,
        "actionable_fix_list": def_issues + str_issues + objective_issues,
        "selection_reason": (
            f"selection-style reward without ground truth: "
            f"exec={exec_score}, solve={solve_score}, obj={obj_score}, def={def_score:.3f}, str={str_score:.3f}, "
            f"semantic={semantic_score:.3f}, semantic_risk={semantic_report.get('overall_semantic_risk')}"
        ),
        "eval_json_valid": True,
        "eval_json_error": None,
        "def_issues": def_issues,
        "str_issues": str_issues,
        "objective_issues": objective_issues,
        "semantic_report": semantic_report,
    }

def evaluate_formulation(formulation, problem_description, api_key=None, code_suggestions=None, objective_history=None):
    """
    Evaluate a formulation by considering completeness, consistency, and LLM judgment
    of the formulation quality based on the problem description and Pyomo execution output.

    Parameters:
    -----------
    formulation : dict
        The formulation to evaluate
    problem_description : str
        Description of the optimization problem
    api_key : str, optional
        OpenAI API key for LLM evaluation. If None, will try to use the
        key from environment variables.

    Returns:
    --------
    float
        A reward value, where higher is better
    """
    import io
    import contextlib
    import pyomo.environ as pyo
    from pyomo.opt import SolverStatus, TerminationCondition

    empty_reasoning = empty_review()

    # Check if the formulation is complete
    if (not formulation['sets'] or
            not formulation['parameters'] or
            not formulation['variables'] or
            formulation['objective'] is None or
            not formulation['constraints']):
        score_report = build_selection_score_report(
            formulation,
            execution_error="incomplete_formulation",
            source="mcts_evaluation",
            problem_description=problem_description,
        )
        return score_report["overall_score"], empty_reasoning, 1, score_report

    # Generate Pyomo code for the formulation
    pyomo_code = generate_pyomo_code(formulation, problem_description, api_key, code_suggestions=code_suggestions)
    if not pyomo_code:
        score_report = build_selection_score_report(
            formulation,
            execution_error="failed_to_generate_pyomo_code",
            source="mcts_evaluation",
            problem_description=problem_description,
        )
        return score_report["overall_score"], empty_reasoning, 1, score_report

    # Execute the Pyomo code and capture output
    execution_output = ""
    execution_error = ""
    try:
        # Capture stdout during execution
        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            # Prepare a safe namespace for execution
            namespace: dict[str, Any] = {
                '__name__': '__main__',
                'pyo': pyo,
                'SolverStatus': SolverStatus,
                'TerminationCondition': TerminationCondition
            }
            # Execute the code
            exec(pyomo_code, namespace)
            main_func = namespace.get("main")
            if f.tell() == 0 and callable(main_func):
                main_func()

        execution_output = f.getvalue()
        pass
    except Exception as e:
        execution_error = str(e)
        execution_output = f"{execution_output}\nError: {execution_error}"

    score_report = build_selection_score_report(
        formulation,
        execution_output=execution_output,
        execution_error=execution_error,
        source="mcts_evaluation",
        problem_description=problem_description,
        pyomo_code=pyomo_code,
    )
    score_report["_pyomo_code"] = pyomo_code
    score_report["_execution_output"] = execution_output
    selection_reward = score_report["overall_score"]
    consensus_report = build_consensus_reward_report(
        score_report["evidence"].get("optimal_value"),
        objective_history=objective_history,
    )

    reasoning = llm_review_candidate(
        formulation=formulation,
        problem_description=problem_description,
        pyomo_code=pyomo_code,
        execution_output=execution_output,
        score_report=score_report,
        code_suggestions=code_suggestions,
        api_key=api_key,
    )
    score_report["_llm_review"] = reasoning
    review_penalty_report = build_review_penalty_report(reasoning)
    score_report.update(review_penalty_report)
    blended_reward = (
        (1.0 - consensus_report["alpha"]) * selection_reward
        + consensus_report["alpha"] * consensus_report["consensus_reward"]
    )
    final_reward = blended_reward * review_penalty_report["review_score"]
    final_reward = apply_review_reward_caps(final_reward, reasoning, score_report)
    score_report["consensus_report"] = consensus_report
    score_report["selection_reward"] = selection_reward
    score_report["blended_reward"] = blended_reward
    score_report["overall_score"] = final_reward
    score_report["selection_reason"] = (
        f"{score_report['selection_reason']}; "
        f"consensus_enabled={consensus_report['enabled']}, "
        f"consensus_alpha={consensus_report['alpha']:.3f}, "
        f"consensus_reward={consensus_report['consensus_reward']:.3f}; "
        f"llm_review_score={review_penalty_report['review_score']:.3f}, "
        f"review_issues={review_penalty_report['review_issues']}"
    )
    uncertainty = 1 - max(0.0, min(1.0, final_reward))

    return final_reward, reasoning, uncertainty, score_report


def _clean_llm_json_content(content):
    """Remove markdown fences and surrounding text from an LLM JSON response.
    
    Args:
        content: Text or object content to write, parse, or normalize.
    
    Returns:
        The value described in the function summary.
    """
    content = str(content or "").strip()
    if content.startswith("```") and "```" in content[3:]:
        first_marker = content.find("```")
        lang_end = content.find("\n", first_marker)
        second_marker = content.rfind("```")
        if lang_end != -1 and second_marker > lang_end:
            content = content[lang_end + 1:second_marker].strip()
            if content.startswith("json"):
                content = content[len("json"):].strip()
            if content.startswith("python"):
                content = content[len("python"):].strip()
    return content


def _normalize_review(review):
    """Normalize a raw LLM review into the expected component-wise review schema.
    
    Args:
        review: Component-wise review dictionary returned by the LLM or rule checks.
    
    Returns:
        The value described in the function summary.
    """
    normalized = empty_review()
    if not isinstance(review, dict):
        return normalized
    if isinstance(review.get("evaluation"), dict):
        review = review["evaluation"]
    if review.get("summary") is not None:
        normalized["review_summary"]["primary_issue"] = str(review.get("summary") or "")
    if isinstance(review.get("review_summary"), dict):
        normalized["review_summary"].update(review["review_summary"])
    if isinstance(review.get("diagnosis"), dict):
        for key, value in review["diagnosis"].items():
            if isinstance(value, dict) and key in normalized["diagnosis"]:
                normalized["diagnosis"][key].update(value)
    normalized["primary_failure_mode"] = str(review.get("primary_failure_mode", "unknown") or "unknown")
    if isinstance(review.get("actionable_fix_list"), list):
        normalized["actionable_fix_list"] = [str(item) for item in review["actionable_fix_list"]]
    audit_value = review.get("numeric_audit")
    if not isinstance(audit_value, dict):
        audit_value = review.get("numeric_coverage_audit")
    if isinstance(audit_value, dict):
        audit = audit_value
        if isinstance(audit.get("missing_or_changed_numbers"), list):
            normalized["numeric_coverage_audit"]["missing_or_changed_numbers"] = [
                str(item) for item in audit.get("missing_or_changed_numbers", []) if str(item).strip()
            ]
        if isinstance(audit.get("unchecked_numeric_groups"), list):
            normalized["numeric_coverage_audit"]["unchecked_numeric_groups"] = [
                str(item) for item in audit.get("unchecked_numeric_groups", []) if str(item).strip()
            ]
        confidence = str(audit.get("confidence", audit.get("coverage_confidence", "Low")) or "Low")
        normalized["numeric_coverage_audit"]["coverage_confidence"] = (
            confidence if confidence in {"High", "Medium", "Low"} else "Low"
        )
    semantic_value = review.get("semantic_flags")
    if not isinstance(semantic_value, dict):
        semantic_value = review.get("semantic_checks")
    if isinstance(semantic_value, dict):
        semantic_checks = semantic_value
        normalized["semantic_checks"]["tags"] = [
            str(item) for item in semantic_checks.get("tags", []) if str(item).strip()
        ] if isinstance(semantic_checks.get("tags"), list) else []
        normalized["semantic_checks"]["issues"] = [
            str(item) for item in semantic_checks.get("issues", []) if str(item).strip()
        ] if isinstance(semantic_checks.get("issues"), list) else []
        for key in (
            "suspicious_zero_solution",
            "placeholder_or_missing_data",
            "answer_metric_mismatch_risk",
            "unit_scale_mismatch_risk",
            "objective_metric_mismatch_risk",
            "missing_business_conversion_factor",
        ):
            normalized["semantic_checks"][key] = 1 if semantic_checks.get(key) in (1, True, "1", "true", "True") else 0
        risk = str(semantic_checks.get("overall_risk", semantic_checks.get("overall_semantic_risk", "low")) or "low").lower()
        normalized["semantic_checks"]["overall_semantic_risk"] = risk if risk in {"low", "medium", "high"} else "low"
    revisions = review.get("revisions")
    revisions = revisions if isinstance(revisions, dict) else {}
    for key in REVIEW_KEYS:
        value = revisions.get(key, review.get(key, {}))
        if not isinstance(value, dict):
            continue
        revise_value = value.get("revise", value.get("need_revise"))
        normalized[key]["need_revise"] = 1 if revise_value in (1, True, "1", "true", "True") else 0
        try:
            normalized[key]["uncertainty"] = float(value.get("uncertainty", 0) or 0)
        except (TypeError, ValueError):
            normalized[key]["uncertainty"] = 0
        normalized[key]["error_type"] = str(value.get("error_type", "") or "")
        normalized[key]["reason"] = str(value.get("reason", "") or "")
        normalized[key]["prompt"] = str(value.get("prompt", "") or "")
    return normalized


def llm_review_candidate(
    formulation,
    problem_description,
    pyomo_code,
    execution_output,
    score_report,
    code_suggestions=None,
    api_key=None,
):
    """Review one MCTS candidate and return layer suggestions."""
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return empty_review()

    api_url = _chat_completions_url()
    model = _llm_model()

    formulation_str = json.dumps(compact_for_prompt(formulation), indent=2, ensure_ascii=False)
    score_report_for_prompt = {
        "overall_score": (score_report or {}).get("overall_score"),
        "score_components": (score_report or {}).get("score_components"),
        "evidence": (score_report or {}).get("evidence"),
        "primary_failure_mode": (score_report or {}).get("primary_failure_mode"),
        "actionable_fix_list": (score_report or {}).get("actionable_fix_list"),
        "selection_reason": (score_report or {}).get("selection_reason"),
        "semantic_report": (score_report or {}).get("semantic_report"),
    }
    score_report_str = json.dumps(compact_for_prompt(score_report_for_prompt), indent=2, ensure_ascii=False)
    code_suggestions_block = format_suggestions_block(
        code_suggestions,
        "Accumulated code-level implementation suggestions from previous iterations:",
    )

    prompt = render_prompt_template(
        "evaluation_prompts.yaml",
        "llm_review_candidate",
        problem_description=problem_description,
        formulation=formulation_str,
        pyomo_code=_truncate_text(pyomo_code, 16000),
        execution_output=_truncate_text(execution_output, 12000),
        score_report=score_report_str,
        code_suggestions_block=code_suggestions_block,
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    start_time = time.time()
    try:
        response = requests.post(api_url, headers=headers, json=data, timeout=LLM_REQUEST_TIMEOUT)
        response.raise_for_status()
        response_json = response.json()
        record_llm_call(
            "llm_review_candidate",
            data,
            response_json,
            api_url=api_url,
            status="success",
            http_status=response.status_code,
            elapsed_seconds=time.time() - start_time,
            call_type="llm_evaluation",
            step="evaluate",
        )
        content = response_json["choices"][0]["message"]["content"].strip()
        parsed = json.loads(_clean_llm_json_content(content))
        review = _normalize_review(parsed)
        return review
    except requests.exceptions.RequestException as error:
        http_status = getattr(getattr(error, "response", None), "status_code", None)
        record_llm_call(
            "llm_review_candidate",
            data,
            api_url=api_url,
            status="request_error",
            http_status=http_status,
            elapsed_seconds=time.time() - start_time,
            error=str(error),
            call_type="llm_evaluation",
            step="evaluate",
        )
        return empty_review()
    except Exception as error:
        record_llm_call(
            "llm_review_candidate",
            data,
            api_url=api_url,
            status="unexpected_error",
            elapsed_seconds=time.time() - start_time,
            error=str(error),
            call_type="llm_evaluation",
            step="evaluate",
        )
        return empty_review()

def _answer_json_prompt_block(answer_schema=None):
    """Render the prompt block that instructs generated code to emit structured answer JSON.
    
    Args:
        answer_schema: Optional schema for the structured answer JSON emitted by generated code.
    
    Returns:
        The value described in the function summary.
    """
    schema_text = json.dumps(normalize_for_json(answer_schema), indent=2, ensure_ascii=False) if answer_schema else "{}"
    return render_prompt_template(
        "evaluation_prompts.yaml",
        "answer_json_contract",
        schema_text=schema_text,
    )


def generate_pyomo_code(formulation, problem_description, api_key, answer_schema=None, code_suggestions=None):
    """
    Generate Pyomo code for the given formulation using LLM

    Parameters:
    -----------
    formulation : dict
        The formulation to generate code for
    problem_description : str
        Description of the optimization problem
    api_key : str
        OpenAI API key

    Returns:
    --------
    str
        Pyomo code as a string
    """
    import os
    import requests
    import json

    # Use provided API key or get from environment
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return ""

    api_url = _chat_completions_url()
    model = _llm_model()

    # Convert formulation to string format
    formulation_str = json.dumps(compact_for_prompt(formulation), indent=2)

    answer_json_instructions = _answer_json_prompt_block(answer_schema)
    code_suggestions_block = format_suggestions_block(
        code_suggestions,
        "Accumulated code-level implementation suggestions from previous reviewed candidates:",
    )

    prompt = render_prompt_template(
        "evaluation_prompts.yaml",
        "generate_pyomo_code",
        problem_description=problem_description,
        formulation=formulation_str,
        code_suggestions_block=code_suggestions_block,
        gurobi_time_limit=GUROBI_TIME_LIMIT_SECONDS,
        answer_json_instructions=answer_json_instructions,
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    start_time = time.time()
    try:
        response = requests.post(api_url, headers=headers, json=data, timeout=LLM_REQUEST_TIMEOUT)
        response.raise_for_status()
        response_json = response.json()
        record_llm_call(
            "generate_pyomo_code",
            data,
            response_json,
            api_url=api_url,
            status="success",
            http_status=response.status_code,
            elapsed_seconds=time.time() - start_time,
            call_type="generate_pyomo_code",
        )

        # Extract the content from the response
        code = response_json['choices'][0]['message']['content'].strip()

        # Clean the response from markdown formatting if present
        if code.startswith("```") and "```" in code[3:]:
            # Find the first and last triple backticks
            first_marker = code.find("```")
            # Skip the language identifier if present
            lang_end = code.find("\n", first_marker)
            # Find the closing backticks
            second_marker = code.find("```", lang_end)
            if second_marker > first_marker:
                # Extract the code without the backticks and language identifier
                code = code[lang_end + 1:second_marker].strip()

        return code
    except requests.exceptions.RequestException as error:
        http_status = getattr(getattr(error, "response", None), "status_code", None)
        record_llm_call(
            "generate_pyomo_code",
            data,
            api_url=api_url,
            status="request_error",
            http_status=http_status,
            elapsed_seconds=time.time() - start_time,
            error=str(error),
            call_type="generate_pyomo_code",
        )
        return ""
    except Exception as error:
        record_llm_call(
            "generate_pyomo_code",
            data,
            api_url=api_url,
            status="unexpected_error",
            elapsed_seconds=time.time() - start_time,
            error=str(error),
            call_type="generate_pyomo_code",
        )
        return ""
    
def generate_pyomo_code_with_error(formulation, problem_description, api_key, previous_pyomo_code, error, answer_schema=None, code_suggestions=None):
    """
    Generate Pyomo code for the given formulation using LLM

    Parameters:
    -----------
    formulation : dict
        The formulation to generate code for
    problem_description : str
        Description of the optimization problem
    api_key : str
        OpenAI API key
    previous_pyomo_code: str
        The wrong pyomo code generated last turn
    error: str
        The error information for last generated pyomo code

    Returns:
    --------
    str
        Pyomo code as a string
    """
    import os
    import requests
    import json

    # Use provided API key or get from environment
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return ""

    api_url = _chat_completions_url()
    model = _llm_model()

    # Convert formulation to string format
    formulation_str = json.dumps(compact_for_prompt(formulation), indent=2)

    answer_json_instructions = _answer_json_prompt_block(answer_schema)
    code_suggestions_block = format_suggestions_block(
        code_suggestions,
        "Accumulated code-level implementation suggestions from previous reviewed candidates:",
    )

    prompt = render_prompt_template(
        "evaluation_prompts.yaml",
        "generate_pyomo_code_with_error",
        problem_description=problem_description,
        formulation=formulation_str,
        code_suggestions_block=code_suggestions_block,
        previous_pyomo_code=previous_pyomo_code,
        error=error,
        gurobi_time_limit=GUROBI_TIME_LIMIT_SECONDS,
        answer_json_instructions=answer_json_instructions,
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    start_time = time.time()
    try:
        response = requests.post(api_url, headers=headers, json=data, timeout=LLM_REQUEST_TIMEOUT)
        response.raise_for_status()
        response_json = response.json()
        record_llm_call(
            "generate_pyomo_code_with_error",
            data,
            response_json,
            api_url=api_url,
            status="success",
            http_status=response.status_code,
            elapsed_seconds=time.time() - start_time,
            call_type="generate_pyomo_code_with_error",
        )

        # Extract the content from the response
        code = response_json['choices'][0]['message']['content'].strip()

        # Clean the response from markdown formatting if present
        if code.startswith("```") and "```" in code[3:]:
            # Find the first and last triple backticks
            first_marker = code.find("```")
            # Skip the language identifier if present
            lang_end = code.find("\n", first_marker)
            # Find the closing backticks
            second_marker = code.find("```", lang_end)
            if second_marker > first_marker:
                # Extract the code without the backticks and language identifier
                code = code[lang_end + 1:second_marker].strip()

        return code
    except requests.exceptions.RequestException as error:
        http_status = getattr(getattr(error, "response", None), "status_code", None)
        record_llm_call(
            "generate_pyomo_code_with_error",
            data,
            api_url=api_url,
            status="request_error",
            http_status=http_status,
            elapsed_seconds=time.time() - start_time,
            error=str(error),
            call_type="generate_pyomo_code_with_error",
        )
        return ""
    except Exception as error:
        record_llm_call(
            "generate_pyomo_code_with_error",
            data,
            api_url=api_url,
            status="unexpected_error",
            elapsed_seconds=time.time() - start_time,
            error=str(error),
            call_type="generate_pyomo_code_with_error",
        )
        return ""

def verify_optimization_solution(
        execution_output: str,
        ground_truth: float,
        api_key: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Uses GPT-4o to verify if the optimization solution matches the ground truth.

    Args:
        execution_output: String containing the output from Pyomo execution
        ground_truth: The expected optimal value (ground truth)
        api_key: OpenAI API key (if None, will try to get from environment variable)

    Returns:
        Tuple of (is_correct: bool, explanation: str)
    """
    # Get API key from environment if not provided
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")

    prompt = render_prompt_template(
        "evaluation_prompts.yaml",
        "verify_optimization_solution",
        execution_output=execution_output,
        ground_truth=ground_truth,
    )

    # Make API call to GPT-4o
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": _llm_model(),
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1
    }
    api_url = _chat_completions_url()
    start_time = time.time()
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=LLM_REQUEST_TIMEOUT)
        response.raise_for_status()
        response_json = response.json()
        record_llm_call(
            "verify_optimization_solution",
            payload,
            response_json,
            api_url=api_url,
            status="success",
            http_status=response.status_code,
            elapsed_seconds=time.time() - start_time,
            call_type="verify_solution",
            step="verify",
        )

        # Extract the response
        result = response_json["choices"][0]["message"]["content"].strip()

        # Determine if the answer is correct based on the response
        is_correct = result.upper().startswith("YES")

        return is_correct, result

    except requests.exceptions.RequestException as e:
        http_status = getattr(getattr(e, "response", None), "status_code", None)
        record_llm_call(
            "verify_optimization_solution",
            payload,
            api_url=api_url,
            status="request_error",
            http_status=http_status,
            elapsed_seconds=time.time() - start_time,
            error=str(e),
            call_type="verify_solution",
            step="verify",
        )
        return False, f"Error in verification: {str(e)}"
    except Exception as e:
        record_llm_call(
            "verify_optimization_solution",
            payload,
            api_url=api_url,
            status="unexpected_error",
            elapsed_seconds=time.time() - start_time,
            error=str(e),
            call_type="verify_solution",
            step="verify",
        )
        return False, f"Error in verification: {str(e)}"

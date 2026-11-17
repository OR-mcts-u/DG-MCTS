import json
import os
import requests
import time

from solver.mcts.node import FormulationNode
from solver.utils import compact_for_prompt, parse_llm_structured_response_with_meta, record_llm_call, render_prompt_template

LLM_REQUEST_TIMEOUT = 120

# Gemini 2.5 Flash: 1,048,576 input tokens; reserve 65,536 for output/thinking.
# At ~4 chars/token the safe input budget is (1,048,576 - 65,536) * 4 ≈ 3,932,160 chars.
_GEMINI_FLASH_MAX_INPUT_CHARS = (1_048_576 - 65_536) * 4


class DynamicLLMComponentGenerator:
    """
    Generate optimization formulation components on demand using LLM API calls
    """

    def __init__(self, api_key=None, model="gpt-4o", base_url=None):
        """Initialize the object and store the state required by later methods.
        
        Args:
            api_key: Configuration key or dictionary key to read.
            model: LLM model name sent in chat-completions requests.
            base_url: OpenAI-compatible API base URL.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("OPENAI_MODEL") or "gpt-4o"
        base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_url = f"{base_url}/chat/completions"
        self.last_call_meta = {}

        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it in the constructor or as OPENAI_API_KEY environment variable.")

    def _cap_suggestions(self, suggestions, max_count=8, max_chars=600):
        """Truncate suggestions list for models with limited context tolerance (e.g. Gemini Flash)."""
        if not self.model.lower().startswith("gemini"):
            return suggestions
        capped = suggestions[-max_count:] if len(suggestions) > max_count else suggestions
        result = []
        for s in capped:
            s = str(s or "").strip()
            result.append(s[:max_chars] + "..." if len(s) > max_chars else s)
        return result

    def generate_next_components(self, problem_description, formulation, tree, level):
        """
        Generate component options for the next level based on the current formulation

        Parameters:
        -----------
        problem_description : str
            Text description of the optimization problem
        formulation: str
            Formulation for the problem
        tree : DynamicMCTS
            Current MCTS tree
        level : int
            The level to generate components for (1=sets, 2=parameters, etc.)

        Returns:
        --------
        list
            List of components for the specified level
        """

        if level == FormulationNode.SETS_LEVEL:
            suggestions = "\n".join([f"{idx}. {item}" for idx, item in enumerate(self._cap_suggestions(tree._dedupe_suggestions(tree.suggestions_for_sets)), 1)])
            return self._generate_sets(problem_description, suggestions)
        elif level == FormulationNode.PARAMETERS_LEVEL:
            suggestions = "\n".join([f"{idx}. {item}" for idx, item in enumerate(self._cap_suggestions(tree._dedupe_suggestions(tree.suggestions_for_parameters)), 1)])
            return self._generate_parameters(problem_description, suggestions, formulation['sets'])
        elif level == FormulationNode.VARIABLES_LEVEL:
            suggestions = "\n".join([f"{idx}. {item}" for idx, item in enumerate(self._cap_suggestions(tree._dedupe_suggestions(tree.suggestions_for_variables)), 1)])
            return self._generate_variables(problem_description, suggestions, formulation['sets'], formulation['parameters'])
        elif level == FormulationNode.OBJECTIVE_LEVEL:
            suggestions = "\n".join([f"{idx}. {item}" for idx, item in enumerate(self._cap_suggestions(tree._dedupe_suggestions(tree.suggestions_for_objective)), 1)])
            return self._generate_objectives(problem_description, suggestions, formulation['sets'],
                                             formulation['parameters'], formulation['variables'])
        elif level == FormulationNode.CONSTRAINTS_LEVEL:
            suggestions = "\n".join([f"{idx}. {item}" for idx, item in enumerate(self._cap_suggestions(tree._dedupe_suggestions(tree.suggestions_for_constraints)), 1)])
            return self._generate_constraints(problem_description, suggestions, formulation['sets'], formulation['parameters'],
                                              formulation['variables'], formulation['objective'])
        else:
            return []

    def generate_full_formulation(self, problem_description):
        """Generate a complete five-element formulation in one LLM call.
        
        Args:
            problem_description: Natural-language optimization problem statement.
        
        Returns:
            The value described in the function summary.
        """
        prompt = render_prompt_template(
            "fast_prompts.yaml",
            "generate_full_formulation",
            problem_description=problem_description,
        )
        response = self._call_llm_api(
            prompt,
            request_label="generate_full_formulation",
            call_type="generate_full_formulation",
            step="generate",
        )
        parsed, parse_meta = parse_llm_structured_response_with_meta(response, dict)
        formulation = self._normalize_full_formulation(parsed)
        self.last_call_meta = {
            **self.last_call_meta,
            "parse_status": parse_meta.get("status"),
            "parse_error": parse_meta.get("error"),
            "parser_used": parse_meta.get("parser_used"),
            "cleaned_response": parse_meta.get("cleaned_response"),
            "raw_response": parse_meta.get("raw_response", self.last_call_meta.get("raw_response", "")),
        }
        return formulation

    def _normalize_full_formulation(self, value):
        """Normalize a raw full-formulation response into the expected five-element dictionary.
        
        Args:
            value: Arbitrary value to normalize, inspect, parse, or serialize.
        
        Returns:
            The value described in the function summary.
        """
        if not isinstance(value, dict):
            return {
                "type": None,
                "sets": [],
                "parameters": [],
                "variables": [],
                "objective": None,
                "constraints": [],
            }

        objective = value.get("objective")
        if isinstance(objective, list):
            objective = objective[0] if objective else None

        return {
            "type": value.get("type"),
            "sets": value.get("sets") if isinstance(value.get("sets"), list) else [],
            "parameters": value.get("parameters") if isinstance(value.get("parameters"), list) else [],
            "variables": value.get("variables") if isinstance(value.get("variables"), list) else [],
            "objective": objective,
            "constraints": value.get("constraints") if isinstance(value.get("constraints"), list) else [],
        }

    def refine_next_components(self, problem_description, formulation, tree, level, revision_requests):
        """Generate revision-conditioned candidate components for the next target layer.
        
        Args:
            problem_description: Natural-language optimization problem statement.
            formulation: Structured five-element optimization formulation.
            tree: Active DynamicMCTS instance providing suggestion pools and context.
            level: Formulation layer index in the MCTS tree.
            revision_requests: Pending diagnostic requests for component-level repair.
        
        Returns:
            The value described in the function summary.
        """
        if level == 0:
            return self.generate_next_components(problem_description, formulation, tree, level)
        return self._refine_component(
            problem_description,
            formulation,
            level,
            revision_requests,
        )

    def _generation_preamble(self):
        """Render the shared prompt preamble for component generation.
        
        Returns:
            The value described in the function summary.
        """
        return render_prompt_template("mcts_prompts.yaml", "generation_preamble")

    def _layer_name(self, level):
        """Map a formulation-tree level number to a human-readable layer name.
        
        Args:
            level: Formulation layer index in the MCTS tree.
        
        Returns:
            The value described in the function summary.
        """
        return {
            FormulationNode.SETS_LEVEL: "SETS",
            FormulationNode.PARAMETERS_LEVEL: "PARAMETERS",
            FormulationNode.VARIABLES_LEVEL: "VARIABLES",
            FormulationNode.OBJECTIVE_LEVEL: "OBJECTIVE",
            FormulationNode.CONSTRAINTS_LEVEL: "CONSTRAINTS",
        }.get(level, "UNKNOWN")

    def _target_output_contract(self, level):
        """Return the prompt contract that defines the expected output for a layer.
        
        Args:
            level: Formulation layer index in the MCTS tree.
        
        Returns:
            The value described in the function summary.
        """
        key = {
            FormulationNode.SETS_LEVEL: "target_output_contract_sets",
            FormulationNode.PARAMETERS_LEVEL: "target_output_contract_parameters",
            FormulationNode.VARIABLES_LEVEL: "target_output_contract_variables",
            FormulationNode.OBJECTIVE_LEVEL: "target_output_contract_objective",
            FormulationNode.CONSTRAINTS_LEVEL: "target_output_contract_constraints",
        }.get(level, "target_output_contract_default")
        return render_prompt_template("mcts_prompts.yaml", key)

    def _layer_responsibility_guidance(self, level):
        """Return prompt guidance describing what a formulation layer is responsible for.
        
        Args:
            level: Formulation layer index in the MCTS tree.
        
        Returns:
            The value described in the function summary.
        """
        key = {
            FormulationNode.SETS_LEVEL: "layer_guidance_sets",
            FormulationNode.PARAMETERS_LEVEL: "layer_guidance_parameters",
            FormulationNode.VARIABLES_LEVEL: "layer_guidance_variables",
            FormulationNode.OBJECTIVE_LEVEL: "layer_guidance_objective",
            FormulationNode.CONSTRAINTS_LEVEL: "layer_guidance_constraints",
        }.get(level)
        return render_prompt_template("mcts_prompts.yaml", key) if key else ""

    def _refine_component(self, problem_description, formulation, level, revision_requests):
        """Ask the LLM to revise one formulation component using diagnostic requests.
        
        Args:
            problem_description: Natural-language optimization problem statement.
            formulation: Structured five-element optimization formulation.
            level: Formulation layer index in the MCTS tree.
            revision_requests: Pending diagnostic requests for component-level repair.
        
        Returns:
            The value described in the function summary.
        """
        layer_name = self._layer_name(level)
        formulation_str = json.dumps(compact_for_prompt(formulation), indent=2, ensure_ascii=False)
        requests_str = json.dumps(compact_for_prompt(revision_requests), indent=2, ensure_ascii=False)
        output_contract = self._target_output_contract(level)
        layer_guidance = self._layer_responsibility_guidance(level)

        prompt = render_prompt_template(
            "mcts_prompts.yaml",
            "refine_component",
            problem_description=problem_description,
            layer_name=layer_name,
            layer_guidance=layer_guidance,
            formulation_json=formulation_str,
            revision_requests_json=requests_str,
            consistency_rules=self._representation_consistency_rules(
                sets=formulation.get("sets"),
                parameters=formulation.get("parameters"),
                variables=formulation.get("variables"),
                objective=formulation.get("objective"),
            ),
            output_contract=output_contract,
        )
        response = self._call_llm_api(
            prompt,
            request_label=f"refine_{layer_name.lower()}",
            call_type="refine_component",
            step="generate",
            level=level,
        )
        parsed = self._parse_llm_response(response)
        if level == FormulationNode.OBJECTIVE_LEVEL:
            return parsed[:1] if parsed else []
        return parsed

    def _representation_consistency_rules(self, sets=None, parameters=None, variables=None, objective=None):
        """Render consistency rules tying sets, parameters, variables, and objective together.
        
        Args:
            sets: Input value used by this function.
            parameters: Input value used by this function.
            variables: Input value used by this function.
            objective: Input value used by this function.
        
        Returns:
            The value described in the function summary.
        """
        context = {
            "sets": sets or [],
            "parameters": parameters or [],
            "variables": variables or [],
            "objective": objective,
        }
        return render_prompt_template(
            "mcts_prompts.yaml",
            "representation_consistency_rules",
            context_json=json.dumps(compact_for_prompt(context), ensure_ascii=False),
        )

    def _generate_sets(self, problem_description, suggestions):
        """Generate set components using LLM"""

        prompt = render_prompt_template(
            "mcts_prompts.yaml",
            "generate_sets",
            problem_description=problem_description,
            generation_preamble=self._generation_preamble(),
            layer_guidance=self._layer_responsibility_guidance(FormulationNode.SETS_LEVEL),
            consistency_rules=self._representation_consistency_rules(),
            suggestions=suggestions,
        )

        response = self._call_llm_api(prompt, request_label="generate_sets", level=FormulationNode.SETS_LEVEL)
        return self._parse_llm_response(response)

    def _generate_parameters(self, problem_description, suggestions, sets):
        """Generate parameter components using LLM"""
        sets_str = str(sets)

        prompt = render_prompt_template(
            "mcts_prompts.yaml",
            "generate_parameters",
            problem_description=problem_description,
            generation_preamble=self._generation_preamble(),
            layer_guidance=self._layer_responsibility_guidance(FormulationNode.PARAMETERS_LEVEL),
            sets=sets_str,
            consistency_rules=self._representation_consistency_rules(sets=sets),
            suggestions=suggestions,
        )

        response = self._call_llm_api(prompt, request_label="generate_parameters", level=FormulationNode.PARAMETERS_LEVEL)
        return self._parse_llm_response(response)

    def _generate_variables(self, problem_description, suggestions, sets, parameters):
        """Generate variable components using LLM"""
        sets_str = str(sets)
        parameters_str = str(parameters)

        prompt = render_prompt_template(
            "mcts_prompts.yaml",
            "generate_variables",
            problem_description=problem_description,
            generation_preamble=self._generation_preamble(),
            layer_guidance=self._layer_responsibility_guidance(FormulationNode.VARIABLES_LEVEL),
            sets=sets_str,
            parameters=parameters_str,
            consistency_rules=self._representation_consistency_rules(sets=sets, parameters=parameters),
            suggestions=suggestions,
        )

        response = self._call_llm_api(prompt, request_label="generate_variables", level=FormulationNode.VARIABLES_LEVEL)
        return self._parse_llm_response(response)

    def _generate_objectives(self, problem_description, suggestions, sets, parameters, variables):
        """Generate objective components using LLM"""
        sets_str = str(sets)
        parameters_str = str(parameters)
        variables_str = str(variables)

        prompt = render_prompt_template(
            "mcts_prompts.yaml",
            "generate_objective",
            problem_description=problem_description,
            generation_preamble=self._generation_preamble(),
            layer_guidance=self._layer_responsibility_guidance(FormulationNode.OBJECTIVE_LEVEL),
            sets=sets_str,
            parameters=parameters_str,
            variables=variables_str,
            consistency_rules=self._representation_consistency_rules(sets=sets, parameters=parameters, variables=variables),
            suggestions=suggestions,
        )


        response = self._call_llm_api(prompt, request_label="generate_objective", level=FormulationNode.OBJECTIVE_LEVEL)
        objectives = self._parse_llm_response(response)

        # Return only the first objective if multiple are generated
        return objectives[:1] if objectives else []

    def _generate_constraints(self, problem_description, suggestions, sets, parameters, variables, objective):
        """Generate constraint components using LLM"""
        sets_str = str(sets)
        parameters_str = str(parameters)
        variables_str = str(variables)
        objective_str = str(objective)

        prompt = render_prompt_template(
            "mcts_prompts.yaml",
            "generate_constraints",
            problem_description=problem_description,
            generation_preamble=self._generation_preamble(),
            layer_guidance=self._layer_responsibility_guidance(FormulationNode.CONSTRAINTS_LEVEL),
            sets=sets_str,
            parameters=parameters_str,
            variables=variables_str,
            objective=objective_str,
            consistency_rules=self._representation_consistency_rules(
                sets=sets,
                parameters=parameters,
                variables=variables,
                objective=objective,
            ),
            suggestions=suggestions,
        )

        response = self._call_llm_api(prompt, request_label="generate_constraints", level=FormulationNode.CONSTRAINTS_LEVEL)
        return self._parse_llm_response(response)

    def _call_llm_api(self, prompt, request_label="llm_request", call_type="generate_component", step="generate", level=None):
        """
        Call the LLM API and get the text response

        Parameters:
        -----------
        prompt : str
            The prompt to send to the LLM

        Returns:
        --------
        str
            Text response from the LLM
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        if self.model.lower().startswith("gemini") and "flash" in self.model.lower():
            if len(prompt) > _GEMINI_FLASH_MAX_INPUT_CHARS:
                prompt = prompt[:_GEMINI_FLASH_MAX_INPUT_CHARS]

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }
        prompt_preview = " ".join(str(prompt).split())[:400]
        start_time = time.time()
        try:
            response = requests.post(self.api_url, headers=headers, json=data, timeout=LLM_REQUEST_TIMEOUT)
            response.raise_for_status()
            response_json = response.json()
            elapsed_seconds = time.time() - start_time
            usage_record = record_llm_call(
                request_label,
                data,
                response_json,
                api_url=self.api_url,
                status="success",
                http_status=response.status_code,
                elapsed_seconds=elapsed_seconds,
                call_type=call_type,
                step=step,
                level=level,
            )

            # Extract the content from the response
            content = response_json['choices'][0]['message']['content'].strip()
            self.last_call_meta = {
                "api_status": "success",
                "http_status": response.status_code,
                "model": self.model,
                "api_url": self.api_url,
                "prompt_preview": prompt_preview,
                "elapsed_seconds": elapsed_seconds,
                "usage_record": usage_record,
                "raw_response": content,
            }
            return content
        except requests.exceptions.RequestException as e:
            elapsed_seconds = time.time() - start_time
            http_status = None
            response_obj = getattr(e, "response", None)
            if response_obj is not None:
                http_status = response_obj.status_code
            usage_record = record_llm_call(
                request_label,
                data,
                api_url=self.api_url,
                status="request_error",
                http_status=http_status,
                elapsed_seconds=elapsed_seconds,
                error=str(e),
                call_type=call_type,
                step=step,
                level=level,
            )
            self.last_call_meta = {
                "api_status": "request_error",
                "http_status": http_status,
                "model": self.model,
                "api_url": self.api_url,
                "prompt_preview": prompt_preview,
                "elapsed_seconds": elapsed_seconds,
                "usage_record": usage_record,
                "error": str(e),
                "raw_response": "",
            }
            return ""
        except Exception as e:
            elapsed_seconds = time.time() - start_time
            usage_record = record_llm_call(
                request_label,
                data,
                api_url=self.api_url,
                status="unexpected_error",
                http_status=None,
                elapsed_seconds=elapsed_seconds,
                error=str(e),
                call_type=call_type,
                step=step,
                level=level,
            )
            self.last_call_meta = {
                "api_status": "unexpected_error",
                "http_status": None,
                "model": self.model,
                "api_url": self.api_url,
                "prompt_preview": prompt_preview,
                "elapsed_seconds": elapsed_seconds,
                "usage_record": usage_record,
                "error": str(e),
                "raw_response": "",
            }
            return ""

    def _parse_llm_response(self, response):
        """
        Parse the LLM response into Python objects

        Parameters:
        -----------
        response : str
            The LLM response

        Returns:
        --------
        list
            List of components parsed from the response
        """
        parsed, parse_meta = parse_llm_structured_response_with_meta(response, list)
        self.last_call_meta = {
            **self.last_call_meta,
            "parse_status": parse_meta.get("status"),
            "parse_error": parse_meta.get("error"),
            "parser_used": parse_meta.get("parser_used"),
            "cleaned_response": parse_meta.get("cleaned_response"),
            "raw_response": parse_meta.get("raw_response", self.last_call_meta.get("raw_response", "")),
        }
        return parsed

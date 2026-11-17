import ast
import json
import os
import re
import time
from contextlib import contextmanager
from functools import lru_cache
from string import Template
from contextlib import redirect_stderr, redirect_stdout

import pyomo.environ as pyo
import yaml
from pyomo.opt import SolverStatus, TerminationCondition


SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(SCRIPT_DIR, "assets")
UNIT_SCALE_FACTORS = (10.0, 100.0, 1000.0, 10000.0)
_LLM_USAGE_CONTEXT = {}
_LLM_USAGE_COUNTERS = {}


@lru_cache(maxsize=None)
def load_prompt_file(filename):
    """Load and cache a YAML prompt-template file from the assets directory.
    
    Args:
        filename: Prompt-template filename under the assets directory.
    
    Returns:
        The value described in the function summary.
    """
    path = os.path.join(ASSETS_DIR, filename)
    with open(path, "r", encoding="utf-8") as file:
        prompts = yaml.safe_load(file) or {}
    if not isinstance(prompts, dict):
        raise ValueError(f"Prompt file must contain a YAML mapping: {path}")
    return prompts


def render_prompt_template(filename, key, **values):
    """Render one named prompt template with string-template substitution.
    
    Args:
        filename: Prompt-template filename under the assets directory.
        key: Configuration key or dictionary key to read.
        values: Sequence of values used for aggregation or matching.
    
    Returns:
        The value described in the function summary.
    """
    prompts = load_prompt_file(filename)
    template = prompts.get(key)
    if template is None:
        raise KeyError(f"Prompt template '{key}' not found in {filename}")
    if not isinstance(template, str):
        raise ValueError(f"Prompt template '{key}' in {filename} must be a string")
    return Template(template).safe_substitute(**values)


def dataset_file(data_path, dataset):
    """Resolve the JSONL file path for a dataset name, including a legacy fallback path.
    
    Args:
        data_path: File or directory path consumed by this helper.
        dataset: Dataset name, normally matching a JSONL file stem.
    
    Returns:
        The value described in the function summary.
    """
    data_new_path = os.path.join(data_path, f"{dataset}.jsonl")
    if os.path.exists(data_new_path):
        return data_new_path

    legacy_path = os.path.join(SCRIPT_DIR, "testset", f"testset_{dataset}.jsonl")
    if os.path.exists(legacy_path):
        return legacy_path

    raise FileNotFoundError(f"Dataset file not found: tried {data_new_path} and {legacy_path}")


def item_problem_text(item):
    """Extract the natural-language problem statement from a dataset row.
    
    Args:
        item: Dataset row or nested value being processed.
    
    Returns:
        The value described in the function summary.
    """
    return item.get("question") or item.get("problem") or item.get("description") or ""


def parse_datasets(dataset_args):
    """Parse comma-separated and space-separated dataset arguments into a flat list.
    
    Args:
        dataset_args: Dataset CLI arguments, possibly comma-separated.
    
    Returns:
        The value described in the function summary.
    """
    datasets = []
    for item in dataset_args:
        datasets.extend(part.strip() for part in item.split(",") if part.strip())
    return datasets or ["1"]


def count_jsonl_records(path):
    """Count non-empty JSONL records in a dataset file.
    
    Args:
        path: File or directory path consumed by this helper.
    
    Returns:
        The value described in the function summary.
    """
    with open(path, "r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def path_safe_name(value):
    """Convert arbitrary text into a filesystem-safe name.
    
    Args:
        value: Arbitrary value to normalize, inspect, parse, or serialize.
    
    Returns:
        The value described in the function summary.
    """
    text = str(value).strip()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return safe or "unknown_model"


@contextmanager
def llm_usage_context(**values):
    """Temporarily attach phase and operation metadata to recorded LLM calls.
    
    Args:
        values: Sequence of values used for aggregation or matching.
    """
    previous = dict(_LLM_USAGE_CONTEXT)
    _LLM_USAGE_CONTEXT.update({key: value for key, value in values.items() if value is not None})
    try:
        yield
    finally:
        _LLM_USAGE_CONTEXT.clear()
        _LLM_USAGE_CONTEXT.update(previous)


def init_llm_usage_file(path):
    """Create or reset the JSONL file used to record LLM calls for one problem.
    
    Args:
        path: File or directory path consumed by this helper.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8"):
        pass
    _LLM_USAGE_COUNTERS[path] = 0


def _request_prompt_chars(request_body):
    """Count prompt characters in a chat-completions request body.
    
    Args:
        request_body: Chat-completions request payload.
    
    Returns:
        The value described in the function summary.
    """
    messages = request_body.get("messages") if isinstance(request_body, dict) else None
    if not isinstance(messages, list):
        return 0
    return sum(len(str((message or {}).get("content", ""))) for message in messages)


def _response_usage(response_json):
    """Extract token-usage fields from a chat-completions response JSON.
    
    Args:
        response_json: Parsed JSON response returned by the LLM endpoint.
    
    Returns:
        The value described in the function summary.
    """
    usage = response_json.get("usage") if isinstance(response_json, dict) else None
    usage = usage if isinstance(usage, dict) else {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def record_llm_call(
    label,
    request_body,
    response_json=None,
    *,
    api_url=None,
    status="success",
    http_status=None,
    elapsed_seconds=None,
    error=None,
    call_type=None,
    step=None,
    phase=None,
    operation=None,
    operation_detail=None,
    iteration=None,
    level=None,
):
    """Append one structured LLM-call record to the active usage JSONL file.
    
    Args:
        label: Input value used by this function.
        request_body: Chat-completions request payload.
        response_json: Parsed JSON response returned by the LLM endpoint.
        api_url: Full chat-completions URL used for the request.
        status: Normalized call status such as success or request_error.
        http_status: Normalized call status such as success or request_error.
        elapsed_seconds: Wall-clock request duration in seconds.
        error: Exception text or diagnostic error message.
        ...: Additional metadata fields forwarded to logs, prompts, or request records.
    
    Returns:
        The value described in the function summary.
    """
    usage_path = os.environ.get("POMS_LLM_USAGE_PATH")
    if not usage_path:
        return None

    context = dict(_LLM_USAGE_CONTEXT)
    usage = _response_usage(response_json or {})
    call_id = _LLM_USAGE_COUNTERS.get(usage_path, 0) + 1
    _LLM_USAGE_COUNTERS[usage_path] = call_id

    record = {
        "call_id": call_id,
        "phase": phase if phase is not None else context.get("phase"),
        "step": step if step is not None else context.get("step"),
        "operation": operation if operation is not None else context.get("operation"),
        "operation_detail": operation_detail if operation_detail is not None else context.get("operation_detail"),
        "call_type": call_type if call_type is not None else context.get("call_type"),
        "label": label,
        "iteration": iteration if iteration is not None else context.get("iteration"),
        "level": level if level is not None else context.get("level"),
        "status": status,
        "http_status": http_status,
        "elapsed_seconds": round(float(elapsed_seconds), 6) if elapsed_seconds is not None else None,
        **usage,
        "model": request_body.get("model") if isinstance(request_body, dict) else None,
        "api_url": api_url,
        "prompt_chars": _request_prompt_chars(request_body),
        "error": error,
        "created_at_epoch": time.time(),
    }

    os.makedirs(os.path.dirname(usage_path), exist_ok=True)
    with open(usage_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def load_llm_usage_records(path):
    """Load LLM-call records from a JSONL usage log.
    
    Args:
        path: File or directory path consumed by this helper.
    
    Returns:
        The value described in the function summary.
    """
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def summarize_llm_usage(records):
    """Aggregate call counts, token totals, elapsed time, and grouped usage statistics.
    
    Args:
        records: Loaded LLM usage records to aggregate.
    
    Returns:
        The value described in the function summary.
    """
    summary = {
        "call_count": len(records),
        "success_count": sum(1 for record in records if record.get("status") == "success"),
        "error_count": sum(1 for record in records if record.get("status") != "success"),
        "elapsed_seconds_total": round(sum(float(record.get("elapsed_seconds") or 0) for record in records), 6),
        "prompt_tokens_total": 0,
        "completion_tokens_total": 0,
        "total_tokens_total": 0,
        "missing_usage_count": 0,
        "by_phase": {},
        "by_phase_step": {},
        "by_operation": {},
        "by_phase_operation": {},
        "by_operation_detail": {},
        "by_call_type": {},
        "by_label": {},
    }

    def add_token_totals(target, record):
        """Run the add_token_totals step in the DG-MCTS solver pipeline.
        
        Args:
            target: Input value used by this function.
            record: Input value used by this function.
        """
        missing = False
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = record.get(key)
            if value is None:
                missing = True
                continue
            target[f"{key}_total"] = target.get(f"{key}_total", 0) + int(value)
        if missing:
            target["missing_usage_count"] = target.get("missing_usage_count", 0) + 1

    def add_group(group, key, record):
        """Run the add_group step in the DG-MCTS solver pipeline.
        
        Args:
            group: Input value used by this function.
            key: Configuration key or dictionary key to read.
            record: Input value used by this function.
        """
        bucket = group.setdefault(key or "unknown", {
            "call_count": 0,
            "success_count": 0,
            "error_count": 0,
            "elapsed_seconds": 0.0,
            "prompt_tokens_total": 0,
            "completion_tokens_total": 0,
            "total_tokens_total": 0,
            "missing_usage_count": 0,
        })
        bucket["call_count"] += 1
        if record.get("status") == "success":
            bucket["success_count"] += 1
        else:
            bucket["error_count"] += 1
        bucket["elapsed_seconds"] = round(bucket["elapsed_seconds"] + float(record.get("elapsed_seconds") or 0), 6)
        add_token_totals(bucket, record)

    for record in records:
        add_token_totals(summary, record)
        phase = record.get("phase") or "unknown"
        step = record.get("step") or "unknown"
        operation = record.get("operation") or "unknown"
        operation_detail = record.get("operation_detail") or operation
        add_group(summary["by_phase"], phase, record)
        add_group(summary["by_phase_step"], f"{phase}.{step}", record)
        add_group(summary["by_operation"], operation, record)
        add_group(summary["by_phase_operation"], f"{phase}.{operation}", record)
        add_group(summary["by_operation_detail"], operation_detail, record)
        add_group(summary["by_call_type"], record.get("call_type"), record)
        add_group(summary["by_label"], record.get("label"), record)

    return summary


def write_text(path, content):
    """Write text to disk, creating parent directories as needed.
    
    Args:
        path: File or directory path consumed by this helper.
        content: Text or object content to write, parse, or normalize.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write("" if content is None else str(content))


def write_json(path, content):
    """Write JSON to disk with stable UTF-8 formatting.
    
    Args:
        path: File or directory path consumed by this helper.
        content: Text or object content to write, parse, or normalize.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(content, file, indent=2, ensure_ascii=False)


def done_file(cache_path, model_name, dataset):
    """Return the resume-cache file path for one model and dataset.
    
    Args:
        cache_path: File or directory path consumed by this helper.
        model_name: Filesystem-safe model name used in output/cache paths.
        dataset: Dataset name, normally matching a JSONL file stem.
    
    Returns:
        The value described in the function summary.
    """
    return os.path.join(cache_path, model_name, dataset, "done.txt")


def load_done_ids(cache_path, model_name, dataset):
    """Load completed problem IDs from the resume-cache file.
    
    Args:
        cache_path: File or directory path consumed by this helper.
        model_name: Filesystem-safe model name used in output/cache paths.
        dataset: Dataset name, normally matching a JSONL file stem.
    
    Returns:
        The value described in the function summary.
    """
    path = done_file(cache_path, model_name, dataset)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        open(path, "a", encoding="utf-8").close()
        return set()

    with open(path, "r", encoding="utf-8") as file:
        return {line.strip() for line in file if line.strip()}


def append_done_id(cache_path, model_name, dataset, problem_id):
    """Append one completed problem ID to the resume-cache file.
    
    Args:
        cache_path: File or directory path consumed by this helper.
        model_name: Filesystem-safe model name used in output/cache paths.
        dataset: Dataset name, normally matching a JSONL file stem.
        problem_id: Dataset problem identifier used for resume tracking.
    """
    path = done_file(cache_path, model_name, dataset)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(f"{problem_id}\n")


def preview_text(value, limit=500):
    """Create a one-line truncated preview for console logging.
    
    Args:
        value: Arbitrary value to normalize, inspect, parse, or serialize.
        limit: Input value used by this function.
    
    Returns:
        The value described in the function summary.
    """
    text = "" if value is None else str(value).replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def extract_numeric_values(value):
    """Extract numeric candidates from scalar values or free-form text.
    
    Args:
        value: Arbitrary value to normalize, inspect, parse, or serialize.
    
    Returns:
        The value described in the function summary.
    """
    if value is None:
        return []
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]

    text = str(value)
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.replace("−", "-")
    text = text.replace("，", "")
    pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    values = []
    for match in re.findall(pattern, text):
        try:
            values.append(float(match))
        except ValueError:
            continue
    return values


def numeric_close(actual, expected, abs_tol=1e-3, rel_tol=1e-4):
    """Compare two numbers with absolute and relative tolerance handling.
    
    Args:
        actual: Numeric value produced by generated code.
        expected: Ground-truth numeric value used for comparison.
        abs_tol: Absolute tolerance for numeric comparison.
        rel_tol: Relative tolerance for numeric comparison.
    
    Returns:
        The value described in the function summary.
    """
    if actual is None or expected is None:
        return False
    if abs(actual) < abs_tol:
        actual = 0.0
    if abs(expected) < abs_tol:
        expected = 0.0
    effective_abs_tol = abs_tol
    if not float(expected).is_integer():
        effective_abs_tol = max(effective_abs_tol, 5e-3)
    return abs(actual - expected) <= max(effective_abs_tol, rel_tol * max(1.0, abs(expected)))


def numeric_match_with_unit_scale(actual, expected, abs_tol=1e-3, rel_tol=1e-4):
    """Compare two numbers directly and under common unit-scale factors.
    
    Args:
        actual: Numeric value produced by generated code.
        expected: Ground-truth numeric value used for comparison.
        abs_tol: Absolute tolerance for numeric comparison.
        rel_tol: Relative tolerance for numeric comparison.
    
    Returns:
        The value described in the function summary.
    """
    if numeric_close(actual, expected, abs_tol=abs_tol, rel_tol=rel_tol):
        return True, "direct", 1.0

    for factor in UNIT_SCALE_FACTORS:
        if numeric_close(actual * factor, expected, abs_tol=abs_tol, rel_tol=rel_tol):
            return True, "actual_times_scale", factor
        if numeric_close(actual, expected * factor, abs_tol=abs_tol, rel_tol=rel_tol):
            return True, "expected_times_scale", factor
    return False, "", 1.0


def extract_answer_json(execution_output):
    """Extract structured answer JSON from generated execution output.
    
    Args:
        execution_output: Captured stdout/stderr from generated solver code.
    
    Returns:
        The value described in the function summary.
    """
    for line in (execution_output or "").splitlines():
        if "__ANSWER_JSON__" not in line:
            continue
        payload = line.split("__ANSWER_JSON__", 1)[1].strip()
        try:
            answer = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(answer, dict):
            return answer
    return None


def extract_answer_values_from_output(execution_output):
    """Extract primary answer or objective values from structured answer JSON.
    
    Args:
        execution_output: Captured stdout/stderr from generated solver code.
    
    Returns:
        The value described in the function summary.
    """
    answer = extract_answer_json(execution_output)
    if not isinstance(answer, dict):
        return []

    for key in ("answer_value", "objective"):
        values = extract_numeric_values(answer.get(key))
        if values:
            return values

    return []


def extract_derived_metric_values_from_output(execution_output):
    """Extract auxiliary numeric metrics from structured answer JSON.
    
    Args:
        execution_output: Captured stdout/stderr from generated solver code.
    
    Returns:
        The value described in the function summary.
    """
    answer = extract_answer_json(execution_output)
    if not isinstance(answer, dict):
        return []

    derived = answer.get("derived_metrics")
    if not isinstance(derived, dict):
        return []

    values = []
    seen = set()
    for value in derived.values():
        for number in extract_numeric_values(value):
            key = round(number, 12)
            if key not in seen:
                seen.add(key)
                values.append(number)
    return values


def extract_objective_values_from_output(execution_output):
    """Extract candidate objective values from solver or generated-code output text.
    
    Args:
        execution_output: Captured stdout/stderr from generated solver code.
    
    Returns:
        The value described in the function summary.
    """
    marker_pattern = r"__RERUN_OBJECTIVES__\s*(\[.*?\])"
    marker_match = re.search(marker_pattern, execution_output or "", flags=re.S)
    if marker_match:
        try:
            values = json.loads(marker_match.group(1))
            parsed = [float(value) for value in values if isinstance(value, (int, float))]
            if parsed:
                return parsed
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

    values = []
    number = r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    patterns = [
        rf"(?i)objective(?:\s+value)?(?:\s*\([^)]*\))?\s*[:=]\s*({number})",
        rf"(?i)best\s+objective\s+({number})",
        rf"(?i)optimal\s+objective(?:\s+value)?(?:\s*\([^)]*\))?\s*[:=]?\s*({number})",
        rf"(?im)^\s*(?:total|minimum|maximum|optimal)\s+(?:profit|cost|revenue|expense|expenses|value|fiber(?:\s+intake)?|fuel(?:\s+expenses)?|makespan)(?:\s*\([^)]*\))?\s*[:=]\s*({number})",
        rf"(?im)^\s*(?:completed\s*products|completedproducts|makespan)\s*[:=]\s*({number})",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, execution_output or ""):
            try:
                values.append(float(str(match).replace(",", "")))
            except ValueError:
                continue
    return values


def verify_with_numeric_first(execution_output, truth, api_key, llm_verifier):
    """Verify a solution using numeric comparison first, falling back to LLM verification only when needed.
    
    Args:
        execution_output: Captured stdout/stderr from generated solver code.
        truth: Ground-truth answer text or numeric value.
        api_key: Configuration key or dictionary key to read.
        llm_verifier: Fallback verifier callable used when numeric checks are inconclusive.
    
    Returns:
        The value described in the function summary.
    """
    answer_values = extract_answer_values_from_output(execution_output)
    objective_values = extract_objective_values_from_output(execution_output)
    derived_metric_values = extract_derived_metric_values_from_output(execution_output)
    truth_values = extract_numeric_values(truth)

    candidate_groups = [
        ("answer_json", answer_values),
        ("objective", objective_values),
        ("derived_metrics", derived_metric_values),
    ]

    if truth_values and any(values for _, values in candidate_groups):
        for source, actual_values in candidate_groups:
            if not actual_values:
                continue
            for actual in actual_values:
                for expected in truth_values:
                    matched, relation, factor = numeric_match_with_unit_scale(actual, expected, rel_tol=1e-4)
                    if matched:
                        if relation == "direct":
                            return True, f"NUMERIC_MATCH source={source} actual={actual}, expected={expected}"
                        return True, (
                            f"UNIT_SCALE_MATCH source={source} actual={actual}, expected={expected}, "
                            f"relation={relation}, scale_factor={factor}"
                        )
        return False, (
            f"NUMERIC_MISMATCH actual_candidates={objective_values[:5] or answer_values[:5]}, "
            f"expected_candidates={truth_values[:5]}"
        )

    try:
        is_correct, explanation = llm_verifier(execution_output, truth, api_key)
        return is_correct, f"LLM_FALLBACK {explanation}"
    except Exception as error:
        return False, (
            f"VERIFICATION_FAILED answer_json={answer_values}, numeric objective={objective_values}, "
            f"derived_metrics={derived_metric_values}, ground_truth={truth_values}, llm_error={error}"
        )


def exec_pyomo_code(pyomo_code, output_buffer):
    """Execute generated Pyomo code in an isolated namespace while capturing stdout and stderr.
    
    Args:
        pyomo_code: Generated Python/Pyomo source code to execute or review.
        output_buffer: String buffer used to capture generated-code output.
    
    Returns:
        The value described in the function summary.
    """
    if not pyomo_code or not str(pyomo_code).strip():
        raise ValueError("Generated Pyomo code is empty.")

    namespace = {
        "__name__": "__main__",
        "pyo": pyo,
        "SolverStatus": SolverStatus,
        "TerminationCondition": TerminationCondition,
        "EXECUTION_OUTPUT_BUFFER": output_buffer,
    }

    with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
        exec(pyomo_code, namespace)
        if output_buffer.tell() == 0:
            run_generated_pyomo_entrypoint(namespace)

    return output_buffer.getvalue()


def run_generated_pyomo_entrypoint(namespace):
    """Run common generated-code entrypoints when the code only defines helpers."""
    if callable(namespace.get("main")):
        namespace["main"]()
        return

    build_model = namespace.get("build_model")
    solve_model = namespace.get("solve_model")
    display_results = namespace.get("display_results")

    if callable(build_model):
        model = build_model()
        namespace.setdefault("model", model)
        results = solve_model(model) if callable(solve_model) else None
        if callable(display_results):
            display_results(model, results)
        return

    model = namespace.get("model")
    solver = namespace.get("solver")
    if model is not None and solver is not None and hasattr(solver, "solve"):
        try:
            solver.options["TimeLimit"] = 300
        except Exception:
            pass
        results = solver.solve(model, tee=True, load_solutions=False)
        namespace.setdefault("results", results)
        termination = getattr(results.solver, "termination_condition", None)
        feasible_terminations = {
            TerminationCondition.optimal,
            TerminationCondition.feasible,
            TerminationCondition.locallyOptimal,
            TerminationCondition.globallyOptimal,
        }
        if termination in feasible_terminations:
            model.solutions.load_from(results)


def normalize_for_json(value):
    """Recursively replace Python objects that are not JSON-serializable."""
    if value is Ellipsis:
        return "..."
    if isinstance(value, dict):
        return {str(key): normalize_for_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [normalize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [normalize_for_json(item) for item in value]
    return value


def compact_for_prompt(value, max_list_items=80, edge_items=12):
    """Return a JSON-safe prompt view with large literal lists summarized."""
    normalized = normalize_for_json(value)

    def is_integer_sequence(items):
        """Run the is_integer_sequence step in the DG-MCTS solver pipeline.
        
        Args:
            items: Sequence of values to normalize, summarize, or de-duplicate.
        
        Returns:
            The value described in the function summary.
        """
        return items and all(isinstance(item, int) and not isinstance(item, bool) for item in items)

    def summarize_list(items):
        """Run the summarize_list step in the DG-MCTS solver pipeline.
        
        Args:
            items: Sequence of values to normalize, summarize, or de-duplicate.
        
        Returns:
            The value described in the function summary.
        """
        if len(items) <= max_list_items:
            return [compact(item) for item in items]

        if is_integer_sequence(items):
            step = items[1] - items[0] if len(items) > 1 else 1
            if step != 0 and all(items[index] - items[index - 1] == step for index in range(1, len(items))):
                stop_exclusive = items[-1] + step
                range_text = (
                    f"range({items[0]}, {stop_exclusive})"
                    if step == 1
                    else f"range({items[0]}, {stop_exclusive}, {step})"
                )
                return {
                    "__compact_list__": "consecutive_integer_range",
                    "range": range_text,
                    "first": items[0],
                    "last": items[-1],
                    "length": len(items),
                    "note": "Prompt summary only; use existing component names and avoid enumerating this list.",
                }

        return {
            "__compact_list__": "truncated",
            "length": len(items),
            "head": [compact(item) for item in items[:edge_items]],
            "tail": [compact(item) for item in items[-edge_items:]],
            "omitted_count": max(0, len(items) - 2 * edge_items),
            "note": "Prompt summary only; the original list is longer.",
        }

    def compact(item):
        """Run the compact step in the DG-MCTS solver pipeline.
        
        Args:
            item: Dataset row or nested value being processed.
        
        Returns:
            The value described in the function summary.
        """
        if isinstance(item, list):
            return summarize_list(item)
        if isinstance(item, tuple):
            return summarize_list(list(item))
        if isinstance(item, dict):
            return {str(key): compact(val) for key, val in item.items()}
        return item

    return compact(normalized)


def _literal_eval_with_numeric_ops(text):
    """Parse Python-like literals while allowing simple numeric expressions like 1/3."""
    tree = ast.parse(text, mode="eval")
    max_range_size = 100_000

    def convert_range_call(node):
        """Run the convert_range_call step in the DG-MCTS solver pipeline.
        
        Args:
            node: MCTS tree node being inspected, expanded, or updated.
        
        Returns:
            The value described in the function summary.
        """
        if not isinstance(node, ast.Call):
            raise ValueError(f"Unsupported expression in LLM structured response: {type(node).__name__}")
        if not isinstance(node.func, ast.Name) or node.func.id != "range":
            raise ValueError("Only range(...) calls are allowed in LLM structured response comprehensions.")
        if node.keywords:
            raise ValueError("Keyword arguments are not allowed in range(...) literals.")

        args = [convert(arg) for arg in node.args]
        if not 1 <= len(args) <= 3 or not all(isinstance(arg, int) for arg in args):
            raise ValueError("range(...) literals must use one to three integer arguments.")
        values = range(*args)
        if len(values) > max_range_size:
            raise ValueError(f"range(...) literal expands to more than {max_range_size} values.")
        return values

    def convert_safe_call(node):
        """Run the convert_safe_call step in the DG-MCTS solver pipeline.
        
        Args:
            node: MCTS tree node being inspected, expanded, or updated.
        
        Returns:
            The value described in the function summary.
        """
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "list"
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.Call)
        ):
            return list(convert_range_call(node.args[0]))
        return list(convert_range_call(node))

    def convert_comprehension_iter(node):
        """Run the convert_comprehension_iter step in the DG-MCTS solver pipeline.
        
        Args:
            node: MCTS tree node being inspected, expanded, or updated.
        
        Returns:
            The value described in the function summary.
        """
        if isinstance(node, ast.Call):
            return list(convert_range_call(node))
        if isinstance(node, (ast.List, ast.Tuple)):
            return [convert(item) for item in node.elts]
        raise ValueError("List comprehension iterables must be literal lists/tuples or range(...).")

    def convert_with_scope(node, scope):
        """Run the convert_with_scope step in the DG-MCTS solver pipeline.
        
        Args:
            node: MCTS tree node being inspected, expanded, or updated.
            scope: Input value used by this function.
        
        Returns:
            The value described in the function summary.
        """
        if isinstance(node, ast.Name):
            if node.id in scope:
                return scope[node.id]
            return node.id
        if isinstance(node, ast.Attribute):
            base = convert_with_scope(node.value, scope)
            return f"{base}.{node.attr}"
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Tuple):
            return tuple(convert_with_scope(item, scope) for item in node.elts)
        if isinstance(node, ast.List):
            return [convert_with_scope(item, scope) for item in node.elts]
        if isinstance(node, ast.Dict):
            return {
                convert_with_scope(key, scope): convert_with_scope(value, scope)
                for key, value in zip(node.keys, node.values)
            }
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = convert_with_scope(node.left, scope)
            right = convert_with_scope(node.right, scope)
            if isinstance(left, str) or isinstance(right, str):
                return f"{left}{right}"
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                return left + right
            if isinstance(left, list) and isinstance(right, list):
                return left + right
        raise ValueError("List comprehension output must be a simple name, tuple, list, dict, constant, or simple addition.")

    def convert_condition(node, scope):
        """Run the convert_condition step in the DG-MCTS solver pipeline.
        
        Args:
            node: MCTS tree node being inspected, expanded, or updated.
            scope: Input value used by this function.
        
        Returns:
            The value described in the function summary.
        """
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            values = [convert_condition(value, scope) for value in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not convert_condition(node.operand, scope)
        if isinstance(node, ast.Compare):
            left = convert_with_scope(node.left, scope)
            for operator, comparator in zip(node.ops, node.comparators):
                right = convert_with_scope(comparator, scope)
                if isinstance(operator, ast.Eq):
                    matched = left == right
                elif isinstance(operator, ast.NotEq):
                    matched = left != right
                elif isinstance(operator, ast.Lt):
                    matched = left < right
                elif isinstance(operator, ast.LtE):
                    matched = left <= right
                elif isinstance(operator, ast.Gt):
                    matched = left > right
                elif isinstance(operator, ast.GtE):
                    matched = left >= right
                elif isinstance(operator, ast.In):
                    if isinstance(right, str) and right.startswith("model."):
                        matched = True
                    else:
                        matched = left in right
                elif isinstance(operator, ast.NotIn):
                    if isinstance(right, str) and right.startswith("model."):
                        matched = True
                    else:
                        matched = left not in right
                else:
                    raise ValueError("Unsupported comparison operator in list comprehension filter.")
                if not matched:
                    return False
                left = right
            return True
        raise ValueError("List comprehension filters must be simple boolean comparisons.")

    def convert_list_comp(node):
        """Run the convert_list_comp step in the DG-MCTS solver pipeline.
        
        Args:
            node: MCTS tree node being inspected, expanded, or updated.
        
        Returns:
            The value described in the function summary.
        """
        if not node.generators:
            raise ValueError("List comprehension must include at least one generator.")
        for generator in node.generators:
            if generator.is_async:
                raise ValueError("Async list comprehensions are not allowed.")
            if not isinstance(generator.target, ast.Name):
                raise ValueError("List comprehension targets must be simple names.")

        results = []

        def expand(generator_index, scope):
            """Run the expand step in the DG-MCTS solver pipeline.
            
            Args:
                generator_index: Sibling index used for deterministic labels.
                scope: Input value used by this function.
            """
            if len(results) > max_range_size:
                raise ValueError(f"List comprehension expands to more than {max_range_size} values.")
            if generator_index == len(node.generators):
                results.append(convert_with_scope(node.elt, scope))
                return

            generator = node.generators[generator_index]
            target_name = generator.target.id
            for value in convert_comprehension_iter(generator.iter):
                next_scope = dict(scope)
                next_scope[target_name] = value
                if any(not convert_condition(condition, next_scope) for condition in generator.ifs):
                    continue
                expand(generator_index + 1, next_scope)

        expand(0, {})
        return results

    def convert(node):
        """Run the convert step in the DG-MCTS solver pipeline.
        
        Args:
            node: MCTS tree node being inspected, expanded, or updated.
        
        Returns:
            The value described in the function summary.
        """
        if isinstance(node, ast.Expression):
            return convert(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.List):
            return [convert(item) for item in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(convert(item) for item in node.elts)
        if isinstance(node, ast.Dict):
            return {convert(key): convert(value) for key, value in zip(node.keys, node.values)}
        if isinstance(node, ast.Call):
            return convert_safe_call(node)
        if isinstance(node, ast.ListComp):
            return convert_list_comp(node)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = convert(node.operand)
            if not isinstance(value, (int, float)):
                raise ValueError("Unary operation is only allowed for numeric values.")
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left = convert(node.left)
            right = convert(node.right)
            if isinstance(node.op, ast.Mult):
                if isinstance(left, (list, tuple)) and isinstance(right, int) and right >= 0:
                    if len(left) * right > max_range_size:
                        raise ValueError(f"Repeated literal expands to more than {max_range_size} values.")
                    return list(left) * right
                if isinstance(right, (list, tuple)) and isinstance(left, int) and left >= 0:
                    if len(right) * left > max_range_size:
                        raise ValueError(f"Repeated literal expands to more than {max_range_size} values.")
                    return left * list(right)
            if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
                raise ValueError("Binary operation is only allowed for numeric values.")
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if right == 0:
                raise ValueError("Division by zero in numeric literal.")
            return left / right
        raise ValueError(f"Unsupported expression in LLM structured response: {type(node).__name__}")

    return convert(tree)


def _repair_mismatched_json_closers(text):
    """Repair simple bracket/brace closer mismatches outside strings."""
    pairs = {"[": "]", "{": "}"}
    openers = set(pairs)
    closers = {"]", "}"}
    repaired = []
    stack = []
    in_string = False
    escape = False

    for char in text:
        if in_string:
            repaired.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            repaired.append(char)
            continue
        if char in openers:
            stack.append(char)
            repaired.append(char)
            continue
        if char in closers:
            if stack:
                expected = pairs[stack[-1]]
                repaired.append(expected)
                stack.pop()
            else:
                repaired.append(char)
            continue
        repaired.append(char)

    while stack:
        repaired.append(pairs[stack.pop()])
    return "".join(repaired)


def _repair_trailing_unterminated_string(text):
    """Repair a missing final quote before trailing list/dict closers."""
    if not text:
        return text

    stripped = text.rstrip()
    suffix = text[len(stripped):]
    trailing_closers = []
    index = len(stripped) - 1
    while index >= 0 and stripped[index] in "]}":
        trailing_closers.append(stripped[index])
        index -= 1

    if not trailing_closers:
        return text

    prefix = stripped[: index + 1]
    closers = stripped[index + 1:]
    quote_char = None
    escape = False
    in_string = False

    for char in prefix:
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if in_string:
            if char == quote_char:
                in_string = False
                quote_char = None
            continue
        if char in ("'", '"'):
            in_string = True
            quote_char = char

    if not in_string or quote_char not in ("'", '"'):
        return text

    if quote_char == "'":
        return f"{prefix}'{closers}{suffix}"
    if quote_char == '"':
        return f'{prefix}"{closers}{suffix}'
    return text


def parse_llm_structured_response_with_meta(response, expected_type):
    """Parse a structured LLM response and return both the parsed result and diagnostics."""
    raw_response = response if isinstance(response, str) else ""
    clean_response = raw_response.strip()
    meta = {
        "status": "unknown",
        "raw_response": raw_response,
        "cleaned_response": clean_response,
        "parser_used": None,
        "error": None,
    }

    if not clean_response:
        meta["status"] = "empty_response"
        meta["error"] = "LLM response was empty."
        return expected_type(), meta

    if "```" in clean_response:
        first_marker = clean_response.find("```")
        if first_marker >= 0:
            second_marker = clean_response.find("```", first_marker + 3)
            if second_marker > first_marker:
                content = clean_response[first_marker + 3:second_marker].strip()
                if content.startswith("json"):
                    content = content[len("json"):].strip()
                if content.startswith("python"):
                    content = content[len("python"):].strip()
                clean_response = content
                meta["cleaned_response"] = clean_response

    parse_candidates = [("original", clean_response)]
    repaired_trailing_string = _repair_trailing_unterminated_string(clean_response)
    if repaired_trailing_string != clean_response:
        parse_candidates.append(("repaired_trailing_unterminated_string", repaired_trailing_string))
    repaired_response = _repair_mismatched_json_closers(clean_response)
    if repaired_response != clean_response:
        parse_candidates.append(("repaired_mismatched_closers", repaired_response))

    parsers = (
        ("json.loads", json.loads),
        ("ast.literal_eval", ast.literal_eval),
        ("literal_eval_with_numeric_ops", _literal_eval_with_numeric_ops),
    )
    last_error = None
    for candidate_name, candidate in parse_candidates:
        for parser_name, parser in parsers:
            try:
                result = parser(candidate)
                result = normalize_for_json(result)
                meta["parser_used"] = (
                    parser_name
                    if candidate_name == "original"
                    else f"{parser_name}:{candidate_name}"
                )
                meta["cleaned_response"] = candidate
                if isinstance(result, expected_type):
                    meta["status"] = "success"
                    return result, meta
                meta["status"] = "type_mismatch"
                meta["error"] = (
                    f"Parsed object type {type(result).__name__} does not match "
                    f"expected {expected_type.__name__}."
                )
                return expected_type(), meta
            except Exception as exc:
                last_error = exc

    meta["status"] = "parse_error"
    meta["error"] = str(last_error)
    return expected_type(), meta

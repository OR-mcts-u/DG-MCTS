"""Frozen pure scoring helpers; no network or solver entry point."""
from typing import Any
import re

def extract_numeric_values(value: Any) -> list[float]:
    if value is None or isinstance(value, bool):
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

def numeric_close(actual: float, expected: float, abs_tol: float = 1e-3, rel_tol: float = 1e-4) -> bool:
    if abs(actual) < abs_tol:
        actual = 0.0
    if abs(expected) < abs_tol:
        expected = 0.0
    effective_abs_tol = abs_tol
    if not float(expected).is_integer():
        effective_abs_tol = max(effective_abs_tol, 5e-3)
    return abs(actual - expected) <= max(effective_abs_tol, rel_tol * max(1.0, abs(expected)))

UNIT_SCALE_FACTORS = (10.0, 100.0, 1000.0, 10000.0)

def numeric_match_with_unit_scale(
    actual: float,
    expected: float,
    abs_tol: float = 1e-3,
    rel_tol: float = 1e-4,
) -> tuple[bool, str, float]:
    if numeric_close(actual, expected, abs_tol=abs_tol, rel_tol=rel_tol):
        return True, "direct", 1.0

    for factor in UNIT_SCALE_FACTORS:
        if numeric_close(actual * factor, expected, abs_tol=abs_tol, rel_tol=rel_tol):
            return True, "actual_times_scale", factor
        if numeric_close(actual, expected * factor, abs_tol=abs_tol, rel_tol=rel_tol):
            return True, "expected_times_scale", factor
    return False, "", 1.0


ERROR_CATEGORY_PATTERNS = {
    "Answer/reporting error": (
        "answer_metric_mismatch",
        "answer_metric_scalar_only",
        "answer_metric_is_scalar_objective_not_decision_vector",
        "answer_vector_missing",
        "decision_vector_question_reported_as_objective",
        "decision_vector_answer_missing_complete_variable_plan",
        "decision_vector_not_returned",
        "decision_vector_not_fully_reported",
        "decision_vector_not_primary_answer",
    ),
    "Code/execution error": (
        "implementation_bug",
        "execution_error",
        "code_execution_error",
        "execution_semantic_failure",
        "runtime",
        "traceback",
        "syntax",
    ),
    "Solver status / feasibility error": (
        "solver_status_issue",
        "solver_status_infeasible",
        "infeasible_model",
        "infeasible_supply_demand_balance",
        "unbounded",
    ),
    "Variable/domain error": (
        "wrong_domain",
        "missing_variable",
        "continuous_count_like",
        "continuous_count_decision",
        "continuous_count_decisions",
        "continuous_domain_for_count_like",
        "integer_domain_relaxation",
        "integer_domain_trap",
    ),
    "Objective modeling error": (
        "wrong_objective_metric",
        "objective_metric_mismatch",
    ),
    "Constraint modeling error": (
        "missing_constraint",
        "wrong_constraint",
        "wrong_constraint_orientation",
        "constraint_contradiction",
        "missing_constraint_or_balance_issue",
        "source_balance_wrong",
        "assignment_cardinality_modeling",
        "one_to_one_assignment_missing",
        "goal_programming_without",
    ),
    "Parameter / numeric coverage error": (
        "missing_parameter",
        "missing_or_changed_numbers",
        "unchecked_numeric_groups",
        "index_mismatch",
        "missing_index_set",
        "missing_set_data",
        "missing_set",
        "placeholder_or_missing_data",
        "placeholder_constraint",
        "unused_cost_parameters",
    ),
    "Unit / business semantics error": (
        "unit_scale_mismatch",
        "missing_business_conversion_factor",
        "business_conversion",
        "ratio_semantics",
        "routing_network_semantics",
        "multi_period_state_transition",
    ),
}

ERROR_CATEGORIES = tuple(ERROR_CATEGORY_PATTERNS)

def normalize_token(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")

def normalize_error_categories(labels: list[str]) -> set[str]:
    normalized = [normalize_token(str(label)) for label in labels]
    categories: set[str] = set()
    for category, patterns in ERROR_CATEGORY_PATTERNS.items():
        if any(pattern in label for label in normalized for pattern in patterns):
            categories.add(category)
    return categories

def close(actual, expected, epsilon=True):
    if actual is None or expected is None:
        return False
    atol = 0.001 if float(expected).is_integer() else 0.005
    tolerance = max(atol, 0.0001 * max(1.0, abs(expected)))
    return abs(actual - expected) <= tolerance + (1e-08 if epsilon else 0)

def compare(actual, expected):
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all((close(a, b, False) for a, b in zip(actual, expected)))
    return close(actual, expected, False)

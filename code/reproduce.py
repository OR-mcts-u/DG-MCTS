"""Recompute the released paper tables from observation-level data, offline.

Run from any directory: python code/reproduce.py
Only derived CSV files in results/ are written. No API or solver is called.
"""
from pathlib import Path
from decimal import Decimal
import csv
import json
import math

from scoring import numeric_match_with_unit_scale, normalize_error_categories, ERROR_CATEGORIES, close, compare

ROOT = Path(__file__).resolve().parents[1]
CORE = {
    'NL4Opt': ['nl4opt'], 'NLP4LP': ['nlp4lp'], 'Mamo-E': ['mamo_easy'],
    'ComplexOR': ['complexor'], 'Mamo-C': ['mamo_complex'],
    'IndustryOR': ['industryor_easy', 'industryor_medium', 'industryor_hard'],
}
MODELS = {'GPT-4o-mini': 'gpt-4o-mini', 'o4-mini': 'o4-mini',
          'Gemini-2.5': 'gemini-2.5-flash', 'GPT-5.4-mini': 'gpt-5.4-mini'}
METHODS = [
    'IO (GPT-4o-mini)', 'IO (Gemini-2.5)', 'IO (o4-mini)', 'IO (GPT-5.4-mini)',
    'CoE (Gemini-2.5)', 'CoE (GPT-5.4-mini)', 'OptiMUS (Gemini-2.5)',
    'OptiMUS (GPT-5.4-mini)', 'ECO (GPT-4o-mini)', 'ECO (o4-mini)',
    'ORLM (Mistral-7B)', 'ORLM (Deepseek-Math-7B)', 'ORLM (LLaMa3-8B)',
    'ORLM (Qwen2.5-7B)', 'LLMOPT (Qwen1.5-14B)',
    *[f'DG-MCTS ({model})' for model in MODELS], 'Experts', 'Senior Undergraduates',
]
EXTENDED = {'Large-Scale-OR': 'large_scale_or', 'Air-NRM (CA)': 'air_nrm_ca',
            'Air-NRM (NP-Flow)': 'air_nrm_np_flow', 'OptiBench': 'optibench',
            'OptMATH-Bench': 'optmath'}
ROUTES = {'Fast Only': 'fast_only', 'DG-MCTS': 'fast_to_slow', 'Always Slow': 'always_slow'}
ROUTE_GROUPS = {'Mamo-E sample': 'mamo_easy', 'Mamo-C sample': 'mamo_complex',
                'NLP4LP sample': 'nlp4lp', 'IndustryOR easy': 'industryor_easy',
                'IndustryOR medium': 'industryor_medium', 'IndustryOR hard': 'industryor_hard',
                'Overall': None}
GROUPS = {'ComplexOR': 'complexor', 'IndustryOR easy': 'industryor_easy',
          'IndustryOR medium': 'industryor_medium', 'IndustryOR hard': 'industryor_hard',
          'Mamo-E': 'mamo_easy', 'Mamo-C': 'mamo_complex', 'NL4Opt': 'nl4opt',
          'NLP4LP': 'nlp4lp', 'Overall': None}


def read(name):
    with (ROOT / 'data' / f'{name}.csv').open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def boolean(value):
    if value in (True, 'True'):
        return True
    if value in (False, 'False'):
        return False
    raise ValueError('Missing or invalid Boolean observation')


def percent(numerator, denominator):
    return f'{100 * numerator / denominator:.1f}' if denominator else ''


def subset(rows, dataset):
    return rows if dataset is None else [r for r in rows if r['dataset'] == dataset]


def numeric_correct(row):
    """Frozen core/reward numeric comparison, including declared unit scales."""
    truth = [float(x) for x in json.loads(row['truth_values'])]
    if not truth:
        return None
    for key in ['answer_values', 'objective_values', 'derived_metric_values']:
        for actual in json.loads(row[key]):
            if any(numeric_match_with_unit_scale(float(actual), expected)[0] for expected in truth):
                return True
    return False


def load_data():
    data = {name: read(name) for name in [
        'core', 'baselines', 'extended', 'routes_accuracy', 'routes_cost',
        'search_diagnostics', 'reference_results', 'reward_targets', 'reward_results', 'reward_baseline',
    ]}
    sizes = dict(core=5816, baselines=5605, extended=1816, routes_accuracy=750,
                 routes_cost=750, search_diagnostics=1454, reward_targets=150, reward_results=1350)
    for name, count in sizes.items():
        assert len(data[name]) == count, (name, len(data[name]))
    inputs = {}
    for path in (ROOT / 'data/datasets').glob('*.jsonl'):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        inputs[path.stem] = {str(r['id']) for r in rows}
        assert len(inputs[path.stem]) == len(rows)
    assert len(inputs) == 13 and sum(map(len, inputs.values())) == 2362
    for name in ['core', 'baselines', 'extended', 'routes_accuracy', 'routes_cost', 'search_diagnostics', 'reward_results']:
        rows = data[name]
        key_fields = (['variant'] if name == 'reward_results' else
                      ['model'] if name == 'core' else
                      ['method'] if name in ['baselines', 'extended', 'routes_accuracy', 'routes_cost'] else [])
        keys = [tuple(r[k] for k in [*key_fields, 'dataset', 'problem_id']) for r in rows]
        assert len(keys) == len(set(keys)), name
        assert all(r['problem_id'] in inputs[r['dataset']] for r in rows), name
    for model in MODELS.values():
        for datasets in CORE.values():
            for dataset in datasets:
                rows = [r for r in data['core'] if r['model'] == model and r['dataset'] == dataset]
                assert {r['problem_id'] for r in rows} == inputs[dataset]
    for row in data['core']:
        assert numeric_correct(row) == boolean(row['numeric_correct'])
    targets = {r['id']: r for r in json.loads((ROOT / 'data/optibench_targets.json').read_text())}
    for row in data['extended']:
        actual, expected = json.loads(row['actual']), json.loads(row['expected'])
        if row['dataset'] == 'optibench':
            target = targets[int(row['problem_id'])]
            assert expected == target['expected'] and json.loads(row['reference_keys']) == target['keys']
            matched = compare(actual, expected)
        else:
            matched = close(actual, expected, False)
        valid = row['solver_status'] not in {'infeasible', 'unbounded', 'infeasibleorunbounded', 'error', 'failed'}
        assert bool(valid and matched) == boolean(row['correct'])
    for row in data['routes_accuracy']:
        assert numeric_correct(row) == boolean(row['correct'])
    for row in data['routes_cost']:
        assert all(math.isfinite(float(row[k])) and float(row[k]) >= 0
                   for k in ['llm_calls', 'tokens', 'wall_seconds'])
    for row in data['search_diagnostics']:
        assert boolean(row['entered_slow']) != boolean(row['fast_accepted'])
        if boolean(row['entered_slow']):
            assert int(row['completed_iterations']) > 0
    for row in data['reward_results']:
        if boolean(row['final_result_available']):
            assert numeric_correct(row) == boolean(row['observed_correct']) == boolean(row['correct'])
        else:
            assert row['observed_correct'] == '' and row['output_state'] == 'missing_final_record'
            assert not boolean(row['correct'])
    missing = [r for r in data['reward_results'] if not boolean(r['final_result_available'])]
    assert [(r['variant'], r['dataset'], r['problem_id']) for r in missing] == [
        ('sem_implementation_heavy', 'industryor_hard', '36')]
    return data


def table2(data):
    references = {(r['method'], r['benchmark']): r['accuracy'] for r in data['reference_results'] if r['table'] == '2'}
    result = []
    for method in METHODS:
        row = dict(method=method)
        for benchmark, datasets in CORE.items():
            if (method, benchmark) in references:
                row[benchmark] = references[method, benchmark]
            else:
                if method.startswith('DG-MCTS ('):
                    model = MODELS[method[len('DG-MCTS ('):-1]]
                    records = [r for r in data['core'] if r['model'] == model and r['dataset'] in datasets]
                    field = 'numeric_correct'
                else:
                    records = [r for r in data['baselines'] if r['method'] == method and r['dataset'] in datasets]
                    field = 'correct'
                assert records, (method, benchmark)
                row[benchmark] = percent(sum(boolean(r[field]) for r in records), len(records)) + '%'
        result.append(row)
    return result


def table3(data):
    references = {r['benchmark']: r for r in data['reference_results'] if r['table'] == '3'}
    result = []
    for benchmark, dataset in EXTENDED.items():
        ref = references[benchmark]
        row = dict(benchmark=benchmark, reference_method=ref['method'], published_accuracy=ref['accuracy'])
        for label, method in [('fast_only', 'fast_only'), ('dg_mcts', 'main')]:
            records = [r for r in data['extended'] if r['dataset'] == dataset and r['method'] == method]
            row['cases'] = len(records)
            row[label + '_correct'] = sum(boolean(r['correct']) for r in records)
            row[label + '_accuracy_percent'] = percent(row[label + '_correct'], len(records))
        result.append(row)
    return result


def table4(data):
    result = []
    for group, dataset in ROUTE_GROUPS.items():
        for method_label, method in ROUTES.items():
            accuracy = subset([r for r in data['routes_accuracy'] if r['method'] == method], dataset)
            costs = subset([r for r in data['routes_cost'] if r['method'] == method], dataset)
            assert {(r['dataset'], r['problem_id']) for r in accuracy} == {(r['dataset'], r['problem_id']) for r in costs}
            n = len(accuracy)
            correct = sum(boolean(r['correct']) for r in accuracy)
            result.append(dict(dataset=group, method=method_label, cases=n, correct=correct,
                               accuracy_percent=percent(correct, n),
                               mean_llm_calls=f"{sum(float(r['llm_calls']) for r in costs) / n:.1f}",
                               mean_tokens=f"{sum(float(r['tokens']) for r in costs) / n:.0f}",
                               mean_wall_seconds=f"{sum(float(r['wall_seconds']) for r in costs) / n:.1f}"))
    return result


def table5(data):
    result = []
    for label, dataset in GROUPS.items():
        rows = subset(data['search_diagnostics'], dataset)
        fast = [r for r in rows if boolean(r['fast_accepted'])]
        slow = [r for r in rows if boolean(r['entered_slow'])]
        fc = sum(boolean(r['correct']) for r in fast)
        sc = sum(boolean(r['correct']) for r in slow)
        result.append(dict(dataset=label, cases=len(rows), fast_accepted=len(fast),
                           acceptance_coverage_percent=percent(len(fast), len(rows)), slow_entered=len(slow),
                           slow_coverage_percent=percent(len(slow), len(rows)), accepted_correct=fc,
                           accepted_accuracy_percent=percent(fc, len(fast)), false_acceptances=len(fast)-fc,
                           slow_correct=sc, slow_accuracy_percent=percent(sc, len(slow)), final_correct=fc+sc,
                           final_accuracy_percent=percent(fc+sc, len(rows))))
    return result


def table6(data):
    outcomes = []
    for label, dataset in GROUPS.items():
        slow = [r for r in subset(data['search_diagnostics'], dataset) if boolean(r['entered_slow'])]
        correct = [r for r in slow if boolean(r['correct'])]
        initial = sum(r['formulation_origin'] == 'initial' for r in correct)
        revised = sum(r['formulation_origin'] == 'revised_new' for r in correct)
        assert initial + revised == len(correct)
        outcomes.append(dict(dataset=label, slow_entered=len(slow), final_correct=len(correct),
                             initial_reused=initial, initial_share_percent=percent(initial, len(correct)),
                             revised_selected=revised, revised_share_percent=percent(revised, len(correct))))
    diagnoses = []
    for category in ERROR_CATEGORIES:
        rows = [r for r in data['search_diagnostics'] if boolean(r['entered_slow']) and
                category in normalize_error_categories(json.loads(r['diagnostic_labels']))]
        correct = sum(boolean(r['correct']) for r in rows)
        diagnoses.append(dict(category=category, detected=len(rows), final_correct=correct,
                              final_incorrect=len(rows)-correct, recovery_percent=percent(correct, len(rows))))
    return outcomes, diagnoses


def reward_sensitivity(data):
    parameters = json.loads((ROOT / 'data/reward_parameters.json').read_text())
    baseline = {r['dataset']: (int(r['correct']), int(r['total'])) for r in data['reward_baseline']}
    result, by_dataset = [], []
    for variant in parameters['variants']:
        for dataset in [*baseline, 'Overall']:
            rows = [r for r in data['reward_results'] if r['variant'] == variant and
                    (dataset == 'Overall' or r['dataset'] == dataset)]
            correct, n = sum(boolean(r['correct']) for r in rows), len(rows)
            bc, bn = (sum(c for c, _ in baseline.values()), sum(n for _, n in baseline.values())) if dataset == 'Overall' else baseline[dataset]
            accuracy, base_accuracy = percent(correct, n), percent(bc, bn)
            row = dict(variant=variant, dataset=dataset, target_cases=n,
                       observed_final_results=sum(boolean(r['final_result_available']) for r in rows),
                       correct=correct, accuracy_percent=accuracy, baseline_correct=bc,
                       baseline_cases=bn, baseline_accuracy_percent=base_accuracy,
                       difference_pp=f'{Decimal(accuracy)-Decimal(base_accuracy):.1f}')
            (result if dataset == 'Overall' else by_dataset).append(row)
    return result, by_dataset


def reproduce(write=True):
    data = load_data()
    outcomes, diagnoses = table6(data)
    reward, reward_by_dataset = reward_sensitivity(data)
    tables = dict(table2_core_accuracy=table2(data), table3_extended_accuracy=table3(data),
                  table4_routes=table4(data), table5_routing=table5(data),
                  table6_search_outcomes=outcomes, table6_diagnosis=diagnoses,
                  reward_sensitivity=reward, reward_by_dataset=reward_by_dataset)
    if write:
        (ROOT / 'results').mkdir(exist_ok=True)
        for name, rows in tables.items():
            with (ROOT / 'results' / f'{name}.csv').open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    return tables


if __name__ == '__main__':
    tables = reproduce()
    print('Recomputed Tables 2–6 and reward sensitivity: eight result CSV files.')

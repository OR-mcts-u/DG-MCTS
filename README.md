<div align="center">

# DG-MCTS

### Deliberation-Gated Monte Carlo Tree Search

**A Hybrid Reasoning Algorithm for Automated Optimization Modeling**

Generate a complete formulation. Evaluate its reliability. Search when further deliberation is needed.

[Method](#method) · [Results](#results) · [Reproduce](#reproduce) · [Code](#code) · [Data](#data)

**6 core benchmarks** &nbsp; | &nbsp; **5 extended benchmarks** &nbsp; | &nbsp; **4 backbone LLMs**

</div>

---

DG-MCTS translates natural-language decision problems into executable optimization models. It combines complete formulation generation with selective, diagnosis-guided search over modeling components. This anonymous first-stage release provides the numerical data behind **Tables 2–6**, the reward-parameter sensitivity results, one reproduction notebook, benchmark inputs, and the core implementation.

## Method

An optimization formulation consists of **sets, parameters, decision variables, an objective, and constraints**. A plausible formulation can still encode the wrong decisions or business relationships, even when its code executes successfully.

DG-MCTS addresses this through three connected steps:

1. **Synthesize and evaluate.** The fast phase generates a complete formulation. Code execution, automatic checks, and structured LLM review produce numerical and diagnostic feedback. This evaluation initializes the search tree and is cached for reuse.
2. **Gate further deliberation.** A programmatic acceptance check and a separate confirmation determine whether the initial candidate can be returned directly. Candidates that do not pass proceed to slow search.
3. **Search and revise.** Monte Carlo tree search reuses the initial formulation and feedback, explores alternative components, and uses diagnostic requests to guide revisions. The final formulation is selected from terminal candidates in the tree.

The paper first evaluates solving accuracy on the core and extended suites, then examines the roles of inference routes, acceptance coverage, initial-formulation reuse, diagnostic recovery, and reward design. The released tables follow that same sequence.

## Results

### Table 2 · Core benchmark suite

Solving accuracy (%) for DG-MCTS with four backbone models. The [complete table](results/table2_core_accuracy.csv) also includes direct, structured, learning-based, and human reference methods.

| Backbone | NL4Opt | NLP4LP | Mamo-E | ComplexOR | Mamo-C | IndustryOR |
|:--|--:|--:|--:|--:|--:|--:|
| GPT-4o-mini | 81.4 | 79.8 | 87.0 | 66.7 | 40.8 | 40.0 |
| o4-mini | 88.3 | 85.5 | 86.3 | **88.9** | 66.8 | 60.0 |
| Gemini-2.5-Flash | **89.6** | 86.0 | 86.2 | 77.8 | 68.7 | **62.0** |
| GPT-5.4-mini | 87.9 | **86.8** | **88.3** | 83.3 | **73.9** | **62.0** |

Bold marks the best DG-MCTS backbone result within each column. Performance depends on both the backbone and the modeling requirements of the benchmark.

### Table 3 · Extended benchmark suite

DG-MCTS and Fast Only use **GPT-5.4-mini**. Published reference values are separate literature comparisons.

| Benchmark | Cases | Reference method | Published (%) | Fast Only (%) | DG-MCTS (%) |
|:--|--:|:--|--:|--:|--:|
| Large-Scale-OR | 101 | LEAN-LLM-OPT | 85.1 | 93.1 | **94.1** |
| Air-NRM (CA) | 15 | LEAN-LLM-OPT | 93.3 | 86.7 | **100.0** |
| Air-NRM (NP-Flow) | 21 | LEAN-LLM-OPT | 92.1 | 85.7 | **95.2** |
| OptiBench | 605 | LLMOPT (Qwen2.5-14B) | 66.4 | 73.2 | **76.7** |
| OptMATH-Bench | 166 | LLMOPT (Qwen2.5-14B) | 40.0 | 36.7 | **42.2** |

DG-MCTS improves upon Fast Only in all five evaluated settings. [Counts and results →](results/table3_extended_accuracy.csv)

### Table 4 · Accuracy and computational cost

The route comparison covers **250 instances**: Mamo-E (50), Mamo-C (50), NLP4LP (50), and IndustryOR easy/medium/hard (40/40/20). Costs are means per instance.

| Inference route | Correct | Accuracy | LLM calls | Tokens | Wall time (s) |
|:--|--:|--:|--:|--:|--:|
| Fast Only | 177/250 | 70.8% | 3.2 | 13,192 | 19.3 |
| **DG-MCTS** | **190/250** | **76.0%** | **62.9** | **259,324** | **305.3** |
| Always Slow | 189/250 | 75.6% | 96.8 | 374,658 | 403.4 |

In this comparison, selective deliberation attains similar accuracy to Always Slow while using fewer calls, tokens, and elapsed seconds. [Dataset-level results →](results/table4_routes.csv)

### Table 5 · Reliability-gated routing

Across the **1,454 core instances**, the gate accepts 670 initial formulations and routes 784 instances to slow search.

| Route | Instances | Coverage | Finally correct | Accuracy within route |
|:--|--:|--:|--:|--:|
| Fast acceptance | 670 | 46.1% | 602 | 89.9% |
| Slow search | 784 | 53.9% | 622 | 79.3% |

Acceptance is a reliability assessment rather than a correctness certificate: 68 accepted cases are incorrect under the recorded evaluation. [Full routing results →](results/table5_routing.csv)

**Table 6** examines what slow search returns. Among its 622 correct final outcomes, **180 (28.9%)** reuse the initial formulation and **442 (71.1%)** select a revised formulation. The released [search outcomes](results/table6_search_outcomes.csv) and [diagnosis results](results/table6_diagnosis.csv) also cover eight diagnostic categories; categories may overlap within an instance.

## Reproduce

Open **[reproduce_results.ipynb](reproduce_results.ipynb)** and run all cells. It reads the observation-level CSV files, checks numerical comparisons, and recomputes Tables 2–6 and reward sensitivity. This is an **offline calculation**: it makes no model or solver calls.

```bash
python -m pip install pandas jupyterlab ipykernel
jupyter lab reproduce_results.ipynb
```

The same aggregation is available with the Python standard library alone:

```bash
python code/reproduce.py
```

The eight derived CSV files are written to `results/`. Percentages use one decimal place; overall costs are instance-weighted means.

## Code

| File or directory | Purpose |
|:--|:--|
| `src/main.py` | Dataset loading, runtime configuration, initial evaluation, and end-to-end execution |
| `src/solver/mcts/search.py` | Tree initialization, selection, expansion, evaluation, revision, and backpropagation |
| `src/solver/mcts/node.py` | Node statistics and the implemented selection score |
| `src/solver/mcts/llm_generator.py` | LLM-based component generation and diagnosis-guided revision |
| `src/solver/evaluation.py` | Automatic evaluation, semantic review, reward composition, and acceptance checks |
| `src/solver/utils.py` | Execution, numerical comparison, dataset adapters, and API utilities |
| `src/assets/` | Prompt templates for synthesis, evaluation, and search |
| `code/reproduce.py` | Offline table aggregation from released observations |
| `code/scoring.py` | Numerical comparison rules and diagnostic-category mapping |

### Run DG-MCTS

Algorithm execution requires an OpenAI-compatible API endpoint, Pyomo, and a usable Gurobi installation and license.

```bash
python -m pip install -r requirements.txt
cp config.example.yaml config.yaml
python src/main.py --config config.yaml --dataset nl4opt --limit 1
```

Set `OPENAI_API_KEY` in your environment and enter your provider URL and model in `config.yaml`. The example defaults to 20 slow-search iterations and three generated component candidates. `--limit 1` runs a single input; remove it to process the selected dataset. Model execution consumes provider credits and produces new outputs, separate from the offline table calculation.

## Data

| Data | Contents |
|:--|:--|
| `data/datasets/` | 13 input JSONL files containing 2,362 benchmark instances |
| `data/core.csv` and `data/baselines.csv` | Per-instance core benchmark outcomes |
| `data/extended.csv` | Per-instance extended benchmark outcomes and comparison targets |
| `data/optibench_targets.json` | Named reference fields, scalar/vector targets, and explicit target derivations |
| `data/routes_accuracy.csv` and `data/routes_cost.csv` | Selected route accuracies and measured computational costs |
| `data/search_diagnostics.csv` | Acceptance decisions, search outcomes, formulation origin, and structured diagnostic labels |
| `data/reference_results.csv` | Inherited and published aggregate comparisons |
| `data/reward_*.csv` and `data/reward_parameters.json` | Reward targets, outcomes, reported baseline, and all tested weights and risk caps |

The core suite has NL4Opt (231), NLP4LP (242), Mamo-E (652), ComplexOR (18), Mamo-C (211), and IndustryOR (100). The five extended datasets add 908 instances. The reward study uses 150 selected targets from these inputs.

**Evaluation definitions.** Table 2 uses the retained numerical comparison rule for DG-MCTS and the recorded evaluator outcomes for local baselines. Tables 5–6 use the saved final-correctness labels in the routing records. Both fields are retained when available; the notebook applies each table's definition. Numerical agreement is an evaluation signal, not a proof of formulation equivalence. OptiBench uses the named targets supplied in `optibench_targets.json`, including explicit vector and derived targets; the other extended datasets use final-objective comparison. Reference-only results remain aggregate values rather than synthetic per-instance records.

**Reward sensitivity.** All five automatic-score weights, six semantic-review weights, and three risk-cap perturbations are retained. The implementation-heavy setting has 149 observed final results; under the declared post-hoc fixed-target convention, its one missing final result is scored unsuccessful, giving **94/150 (62.7%)** overall and **11/20 (55.0%)** on IndustryOR hard. The missing observation remains blank in `observed_correct` and is distinguished from an observed empty output. Comparisons with the separately reported baseline are descriptive aggregate differences. [Reward results →](results/reward_sensitivity.csv)

## License

The project code is provided under the [MIT License](LICENSE). Third-party benchmark content remains subject to its original attribution and usage terms.

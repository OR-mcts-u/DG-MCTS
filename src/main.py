# -*- coding: utf-8 -*-
import time
import io
import os
import jsonlines
import argparse
import yaml
from solver.mcts import DynamicMCTS, DynamicLLMComponentGenerator
from solver.evaluation import (
    build_selection_score_report,
    evaluate_formulation,
    formulation_to_5element_description,
    generate_pyomo_code,
    generate_pyomo_code_with_error,
    objective_direction_from_formulation,
    verify_optimization_solution,
)
from solver.utils import (
    append_done_id,
    count_jsonl_records,
    dataset_file,
    done_file,
    exec_pyomo_code,
    item_problem_text,
    init_llm_usage_file,
    load_llm_usage_records,
    llm_usage_context,
    load_done_ids,
    parse_datasets,
    path_safe_name,
    preview_text,
    summarize_llm_usage,
    verify_with_numeric_first,
    write_json,
    write_text,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(SCRIPT_DIR, "data")
DEFAULT_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "output")
DEFAULT_CACHE_PATH = os.path.join(SCRIPT_DIR, "cache")
DEFAULT_USAGE_PATH = os.path.join(SCRIPT_DIR, "usage_analysis")
DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.yaml")
DEFAULT_DATASETS = [
    "complexor",
    "industryor_easy",
    "industryor_medium",
    "industryor_hard",
    "mamo_easy",
    "mamo_complex",
    "nl4opt",
    "nlp4lp",
]

DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "")

DEFAULT_BASE_URL = ""
DEFAULT_MODEL = ""


def load_run_config(path):
    """Load the YAML runtime configuration file and return both its mapping and absolute path.
    
    Args:
        path: File or directory path consumed by this helper.
    
    Returns:
        The value described in the function summary.
    """
    if not path:
        return {}, ""
    if not os.path.exists(path):
        return {}, os.path.abspath(path)
    config_path = os.path.abspath(path)
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return config, config_path


def config_value(config, key, default=None):
    """Read one value from the runtime configuration while preserving a supplied default for missing or null keys.
    
    Args:
        config: Runtime configuration mapping loaded from YAML.
        key: Configuration key or dictionary key to read.
        default: Fallback value used when the key is absent or null.
    
    Returns:
        The value described in the function summary.
    """
    value = config.get(key, default)
    if value is None:
        return default
    return value


def config_path_value(config, config_path, key, default):
    """Resolve a path-valued configuration entry relative to the config file location when it is not absolute.
    
    Args:
        config: Runtime configuration mapping loaded from YAML.
        config_path: File or directory path consumed by this helper.
        key: Configuration key or dictionary key to read.
        default: Fallback value used when the key is absent or null.
    
    Returns:
        The value described in the function summary.
    """
    value = config_value(config, key, default)
    if not value or os.path.isabs(str(value)):
        return value
    base_dir = os.path.dirname(config_path) if config_path else SCRIPT_DIR
    return os.path.join(base_dir, str(value))


def parse_args():
    """Parse command-line arguments after loading YAML defaults, letting explicit CLI values override the config file.
    
    Returns:
        The value described in the function summary.
    """
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, type=str, help="YAML runtime config file")
    bootstrap_args, _ = bootstrap_parser.parse_known_args()
    config, config_path = load_run_config(bootstrap_args.config)

    parser = argparse.ArgumentParser(
        description="Command line interface for POMS LLM Solver.",
        parents=[bootstrap_parser],
    )
    parser.add_argument("--dataset", default=config_value(config, "dataset", config_value(config, "datasets", DEFAULT_DATASETS)), nargs="+", help="choose one or more datasets")
    parser.add_argument("--data_path", default=config_path_value(config, config_path, "data_path", DEFAULT_DATA_PATH), type=str, help="directory containing dataset jsonl files")
    parser.add_argument("--output_path", default=config_path_value(config, config_path, "output_path", DEFAULT_OUTPUT_PATH), type=str, help="output root directory")
    parser.add_argument("--cache_path", default=config_path_value(config, config_path, "cache_path", DEFAULT_CACHE_PATH), type=str, help="cache root for breakpoint resume records")
    parser.add_argument("--usage_path", default=config_path_value(config, config_path, "usage_path", DEFAULT_USAGE_PATH), type=str, help="usage-analysis output root")
    parser.add_argument("--output_file", default=config_value(config, "output_file", "verification_results"), type=str)
    parser.add_argument("--openAI_key", default=config_value(config, "openAI_key", config_value(config, "api_key", DEFAULT_API_KEY)), type=str)
    parser.add_argument("--api_key_env", default=config_value(config, "api_key_env", "OPENAI_API_KEY"), type=str, help="environment variable used when --openAI_key is not provided")
    parser.add_argument("--base_url", default=config_value(config, "base_url", DEFAULT_BASE_URL), type=str)
    parser.add_argument("--LLM_model", default=config_value(config, "LLM_model", config_value(config, "model", DEFAULT_MODEL)), type=str)
    parser.add_argument("--num_components", default=config_value(config, "num_components", 3), type=int)
    parser.add_argument("--iter", default=config_value(config, "iter", config_value(config, "iterations", 20)), type=int)
    parser.add_argument("--time_limit", default=config_value(config, "time_limit", 500), type=int)
    parser.add_argument("--initial_accept_threshold", default=config_value(config, "initial_accept_threshold", 1.0), type=float, help="early accept threshold for the initial rollout; 0 disables it")
    parser.add_argument("--flag", default=config_value(config, "flag", 0), type=int, help="problem id to run; 0 means run all selected items")
    parser.add_argument("--start", default=config_value(config, "start", 0), type=int, help="start index in dataset, 0-based")
    parser.add_argument("--limit", default=config_value(config, "limit", 0), type=int, help="maximum number of problems to run; 0 means all")

    args = parser.parse_args()
    args.config = config_path or os.path.abspath(bootstrap_args.config)
    if not args.openAI_key and args.api_key_env:
        args.openAI_key = os.environ.get(args.api_key_env, "")
    return args


def run_dynamic_mcts(
    problem_text,
    api_key,
    model=DEFAULT_MODEL,
    base_url=DEFAULT_BASE_URL,
    iterations=50,
    time_limit=500,
    num_components=3,
    initial_accept_threshold=0.0,
):
    """
    Run dynamic MCTS with LLM component generation to solve an optimization problem

    Parameters:
    -----------
    problem_text : str
        Text description of the optimization problem
    api_key : str
        OpenAI API key
    iterations : int
        Maximum number of MCTS iterations
    time_limit : int
        Maximum time in seconds for MCTS search

    Returns:
    --------
    tuple
        (best_formulation, pyomo_code)
    """
    # Set API key in environment
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_MODEL"] = model
    os.environ["OPENAI_BASE_URL"] = base_url

    # Create LLM component generator
    llm_generator = DynamicLLMComponentGenerator(api_key=api_key, model=model, base_url=base_url)

    # Run MCTS
    mcts = DynamicMCTS(
        problem_text,
        llm_generator,
        evaluate_formulation,
        initial_accept_threshold=initial_accept_threshold,
    )
    # sys.stdout = io.StringIO()
    start_time = time.time()
    best_formulation = mcts.search(iterations=iterations, time_limit=time_limit, num_components=num_components)
    end_time = time.time()
    # sys.stdout = sys.__stdout__

    # Generate Pyomo code for the best formulation
    cached_report = mcts.best_score_report or {}
    cached_pyomo_code = cached_report.get("_pyomo_code")
    if cached_pyomo_code:
        pyomo_code = cached_pyomo_code
    else:
        with llm_usage_context(
            phase="postprocess",
            step="generate",
            operation="postprocess_generation",
            operation_detail="final_code_generation",
        ):
            pyomo_code = generate_pyomo_code(
                best_formulation,
                problem_text,
                api_key,
                code_suggestions=mcts.unique_code_suggestions(),
            )

    return best_formulation, pyomo_code, end_time - start_time, mcts


def run_case(
    problem_text,
    api_key,
    model=DEFAULT_MODEL,
    base_url=DEFAULT_BASE_URL,
    iterations=50,
    time_limit=500,
    num_components=3,
    max_retry_time=5,
    initial_accept_threshold=0.0,
):
    # Run MCTS with LLM component generation
    """Run one problem instance end to end: search for a formulation, generate code, execute it, and repair code on failures.
    
    Args:
        problem_text: Natural-language optimization problem statement.
        api_key: Configuration key or dictionary key to read.
        model: LLM model name sent in chat-completions requests.
        base_url: OpenAI-compatible API base URL.
        iterations: Maximum number of MCTS iterations for one problem.
        time_limit: Maximum search time in seconds for one problem.
        num_components: Number of candidate components generated per expansion.
        max_retry_time: Maximum number of code-regeneration attempts after execution errors.
        ...: Additional metadata fields forwarded to logs, prompts, or request records.
    
    Returns:
        The value described in the function summary.
    """
    formulation, pyomo_code, running_time, mcts = run_dynamic_mcts(
        problem_text,
        api_key,
        model=model,
        base_url=base_url,
        iterations=iterations,
        time_limit=time_limit,
        num_components=num_components,
        initial_accept_threshold=initial_accept_threshold,
    )
    # Create a string buffer to capture output
    output_buffer = io.StringIO()

    # Execute the generated code and capture its output
    cnt_error_propagation = 0
    pyomo_log = ""
    # If error, continue generate new pyomo code until there is no error or exceed max_retry_time
    try:
        execution_output = exec_pyomo_code(pyomo_code, output_buffer)
        pyomo_log = execution_output

    except Exception as error:
        
        # Handle pyomo error
        while True:
            if cnt_error_propagation > max_retry_time:
                execution_output = f"The maximum number of attempts to regenerate the pyomo code has been reached. The last error is {error}."
                pyomo_log = output_buffer.getvalue() + "\n" + execution_output
                break
            try:
                cnt_error_propagation += 1
                output_buffer = io.StringIO()
                with llm_usage_context(
                    phase="postprocess",
                    step="generate",
                    operation="postprocess_generation",
                    operation_detail="code_repair_generation",
                ):
                    pyomo_code = generate_pyomo_code_with_error(
                        formulation,
                        problem_text,
                        api_key,
                        pyomo_code,
                        error,
                        code_suggestions=mcts.unique_code_suggestions(),
                    )
                execution_output = exec_pyomo_code(pyomo_code, output_buffer)
                pyomo_log = execution_output
            except Exception as e:
                error = e
                time.sleep(3)
                continue
            else:
                break

    return formulation, pyomo_code, execution_output, pyomo_log, cnt_error_propagation, running_time, mcts


# Example usage
if __name__ == "__main__":

    args = parse_args()
    if not args.openAI_key:
        raise SystemExit(f"OpenAI API key is required. Set {args.api_key_env} or pass --openAI_key.")
    if not args.LLM_model:
        raise SystemExit("LLM model is required. Set LLM_model in config.yaml or pass --LLM_model.")

    os.environ["OPENAI_API_KEY"] = args.openAI_key
    os.environ["OPENAI_MODEL"] = args.LLM_model
    os.environ["OPENAI_BASE_URL"] = args.base_url
    model_output_name = path_safe_name(args.LLM_model)
    output_root = os.path.join(args.output_path, model_output_name)
    cache_root = os.path.join(args.cache_path, model_output_name)
    datasets = parse_datasets(args.dataset)

    print("=" * 72)
    print("POMS LLM Solver Run")
    print("=" * 72)
    print(f"[Config] config_path={args.config}")
    print(f"[Config] model={args.LLM_model}")
    print(f"[Config] model_output_name={model_output_name}")
    print(f"[Config] base_url={args.base_url}")
    print(f"[Config] datasets={', '.join(datasets)}")
    print(f"[Config] data_path={args.data_path}")
    print(f"[Config] output_root={output_root}")
    print(f"[Config] cache_root={cache_root}")
    print(f"[Config] usage_root={os.path.join(args.usage_path, model_output_name)}")
    print(f"[Config] iter={args.iter}, time_limit={args.time_limit}, num_components={args.num_components}")
    print(f"[Config] initial_accept_threshold={args.initial_accept_threshold}")
    print(f"[Config] start={args.start}, limit={args.limit}, flag={args.flag}")

    for dataset in datasets:
        Final_result = []
        Final_explain = []
        # read the id, ground truth, and problem text
        flag = args.flag
        dataset_path = dataset_file(args.data_path, dataset)
        dataset_total = count_jsonl_records(dataset_path)
        output_dir = os.path.join(output_root, dataset)
        os.makedirs(output_dir, exist_ok=True)
        write_json(os.path.join(output_dir, "run_config.json"), {
            "dataset": dataset,
            "datasets": datasets,
            "config_path": args.config,
            "data_path": dataset_path,
            "model": args.LLM_model,
            "model_output_name": model_output_name,
            "base_url": args.base_url,
            "api_key_env": args.api_key_env,
            "iterations": args.iter,
            "time_limit": args.time_limit,
            "num_components": args.num_components,
            "initial_accept_threshold": args.initial_accept_threshold,
            "flag": args.flag,
            "start": args.start,
            "limit": args.limit,
            "output_path": output_dir,
            "cache_path": os.path.join(cache_root, dataset),
        })
        done_ids = load_done_ids(args.cache_path, model_output_name, dataset)
        print("\n" + "=" * 72)
        print(f"[Dataset] {dataset}")
        print("=" * 72)
        print(f"[Dataset file] {dataset_path}")
        print(f"[Dataset size] {dataset_total} records")
        print(f"[Output] {output_dir}")
        print(f"[Run config] {os.path.join(output_dir, 'run_config.json')}")
        print(f"[Resume cache] {done_file(args.cache_path, model_output_name, dataset)}")
        print(f"[Resume cache] {len(done_ids)} done problem ids loaded")

        with jsonlines.open(dataset_path) as reader:
            processed = 0
            for index, item in enumerate(reader):
                if index < args.start:
                    continue
                if args.limit and processed >= args.limit:
                    break
                id = item.get('id')
                if flag and id != flag:
                    continue
                id_key = str(id)
                if id_key in done_ids:
                    print(f"[Skip done] dataset={dataset}, index={index}, id={id}")
                    continue

                ground_truth = item.get('ground_truth')
                problem_text = item_problem_text(item)
                print("\n" + "-" * 72)
                print(f"[Problem] dataset={dataset}, index={index}, id={id}")
                print(f"[Progress] processed_this_run={processed}, dataset_records={dataset_total}")
                print(f"[Ground Truth] {preview_text(ground_truth, 300)}")
                print(f"[Problem Text] {preview_text(problem_text, 900)}")
                print("-" * 72)
                start_time = time.time()
                problem_dir = os.path.join(output_dir, "problem_{}".format(id))
                usage_dir = os.path.join(args.usage_path, model_output_name, dataset, "problem_{}".format(id))
                usage_calls_path = os.path.join(usage_dir, "llm_calls.jsonl")
                init_llm_usage_file(usage_calls_path)
                os.environ["POMS_LLM_USAGE_PATH"] = usage_calls_path
                formulation, pyomo_code, execution_output, pyomo_log, cnt_error_propagation, running_time, mcts = run_case(
                    problem_text,
                    args.openAI_key,
                    model=args.LLM_model,
                    base_url=args.base_url,
                    iterations=args.iter,
                    time_limit=args.time_limit,
                    num_components=args.num_components,
                    initial_accept_threshold=args.initial_accept_threshold,
                )
                
                with llm_usage_context(
                    phase="postprocess",
                    step="verify",
                    operation="verification",
                    operation_detail="final_answer_verification",
                ):
                    is_correct, explanation = verify_with_numeric_first(
                        execution_output,
                        ground_truth,
                        args.openAI_key,
                        verify_optimization_solution,
                    )
                wall_time = time.time() - start_time
                llm_usage_records = load_llm_usage_records(usage_calls_path)
                llm_usage_summary = summarize_llm_usage(llm_usage_records)
                postprocess_seconds = max(
                    0.0,
                    wall_time
                    - float((getattr(mcts, "phase_timing", {}) or {}).get("fast_seconds") or 0)
                    - float((getattr(mcts, "phase_timing", {}) or {}).get("slow_seconds") or 0),
                )
                phase_timing = {
                    "fast_seconds": round(float((getattr(mcts, "phase_timing", {}) or {}).get("fast_seconds") or 0), 6),
                    "slow_seconds": round(float((getattr(mcts, "phase_timing", {}) or {}).get("slow_seconds") or 0), 6),
                    "postprocess_seconds": round(postprocess_seconds, 6),
                    "wall_seconds": round(wall_time, 6),
                }
                summary = {
                    'id': id,
                    'dataset': dataset,
                    'correct': is_correct,
                    'running_time': running_time,
                    'wall_time': wall_time,
                    'phase_timing': phase_timing,
                    'llm_usage': llm_usage_summary,
                    'llm_usage_dir': usage_dir,
                    'llm_calls_path': usage_calls_path,
                    'code_generate_time': cnt_error_propagation + 1,
                    'verification': explanation,
                    'explanation': explanation,
                }
                cached_score_report = mcts.best_score_report or {}
                score_report = build_selection_score_report(
                    formulation,
                    execution_output=execution_output,
                    execution_error="" if "Error:" not in str(execution_output) else str(execution_output),
                    source="mcts_final",
                    problem_description=problem_text,
                    pyomo_code=pyomo_code,
                )
                if cached_score_report:
                    score_report["cached_mcts_score_report"] = cached_score_report
                    score_report["evidence"] = dict(score_report.get("evidence") or {})
                    score_report["evidence"]["cached_mcts_score_available"] = True
                final_execution_score = score_report["overall_score"]
                final_score_components = score_report["score_components"]
                mcts_selection_score = cached_score_report.get("overall_score") if cached_score_report else None
                mcts_score_components = cached_score_report.get("score_components") if cached_score_report else None
                if mcts_selection_score is not None:
                    score_report["mcts_selection_score"] = mcts_selection_score
                    score_report["mcts_score_components"] = mcts_score_components
                    score_report["final_execution_score"] = final_execution_score
                    score_report["final_score_components"] = final_score_components
                    summary["selection_score"] = mcts_selection_score
                    summary["score_components"] = mcts_score_components
                    summary["final_execution_score"] = final_execution_score
                    summary["final_score_components"] = final_score_components
                else:
                    score_report["final_execution_score"] = final_execution_score
                    score_report["final_score_components"] = final_score_components
                    summary["selection_score"] = final_execution_score
                    summary["score_components"] = final_score_components
                Final_result.append(is_correct)
                Final_explain.append(summary)
                processed += 1
                write_text(os.path.join(problem_dir, "problem_description.txt"), problem_text)
                write_text(os.path.join(problem_dir, "ground_truth.txt"), ground_truth)
                write_json(os.path.join(problem_dir, "best_formulation.json"), formulation)
                write_text(os.path.join(problem_dir, "objective_direction.txt"), objective_direction_from_formulation(formulation))
                write_text(os.path.join(problem_dir, "5element_description.txt"), formulation_to_5element_description(formulation))
                write_json(os.path.join(problem_dir, "score.json"), score_report)
                write_text(os.path.join(problem_dir, "final_pyomo_code.py"), pyomo_code)
                write_text(os.path.join(problem_dir, "execution_output.txt"), execution_output)
                write_text(os.path.join(problem_dir, "pyomo_log.txt"), pyomo_log)
                write_text(os.path.join(problem_dir, "mcts_operation_log.txt"), mcts.operation_log_text())
                write_text(os.path.join(problem_dir, "mcts_tree.md"), mcts.tree_markdown())
                write_json(os.path.join(problem_dir, "initial_simulate_review.json"), mcts.initial_simulate_review or {})
                write_json(os.path.join(problem_dir, "initial_simulate_check.json"), mcts.initial_simulate_check or {})
                write_json(os.path.join(problem_dir, "best_result_review.json"), mcts.best_review or {})
                write_json(os.path.join(problem_dir, "suggestions.json"), {
                    "deduped_for_prompt": mcts.suggestions_snapshot(),
                    "raw_accumulated": mcts.raw_suggestions_snapshot(),
                })
                write_json(os.path.join(problem_dir, "summary.json"), summary)
                write_json(os.path.join(usage_dir, "usage_summary.json"), {
                    "id": id,
                    "dataset": dataset,
                    "model": args.LLM_model,
                    "phase_timing": phase_timing,
                    "llm_usage": llm_usage_summary,
                    "llm_calls_path": usage_calls_path,
                })
                write_text(os.path.join(problem_dir, "verification.txt"), explanation)
                # Save the results to a JSONl file
                output_file = os.path.join(output_dir, args.output_file+"_["+str(flag) + ", " + str(args.num_components) + ", " + str(args.iter) + "].json")
                with jsonlines.open(output_file, mode="w") as writer:
                    writer.write_all(Final_explain)
                append_done_id(args.cache_path, model_output_name, dataset, id_key)
                done_ids.add(id_key)
                print(f"[Result] dataset={dataset}, id={id}, correct={is_correct}")
                print(f"[Score] selection_score={summary['selection_score']}, final_execution_score={summary.get('final_execution_score')}, components={summary['score_components']}")
                print(f"[Verification] {explanation}")
                print(f"[Timing] mcts={running_time:.2f}s, wall={wall_time:.2f}s, code_generations={cnt_error_propagation + 1}")
                print(f"[Usage] calls={llm_usage_summary['call_count']}, total_tokens={llm_usage_summary['total_tokens_total']}, usage_dir={usage_dir}")
                print(f"[Saved] {problem_dir}")
                print(f"[Running accuracy] {100 * sum(Final_result) / len(Final_result):.2f}% ({sum(Final_result)}/{len(Final_result)})")

        if Final_result:
            print("\n" + "-" * 72)
            print(f"[Dataset summary] {dataset}: accuracy={100 * sum(Final_result) / len(Final_result):.2f}% ({sum(Final_result)}/{len(Final_result)})")
            print(f"[Dataset summary] output={output_dir}")
        else:
            print("\n" + "-" * 72)
            print(f"[Dataset summary] {dataset}: no new problems processed")

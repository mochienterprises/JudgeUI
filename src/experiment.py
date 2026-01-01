"""Experiment runner with matrix expansion and parallel execution."""

import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Callable

import yaml

from .config import load_config, get_provider_for_model, CONFIG_DIR
from .evaluator import evaluate_argument
from .models import Argument, ExperimentConfig, ExperimentResults, EvaluationResult
from .storage import load_argument, load_arguments, save_results, append_evaluation, ARGUMENTS_DIR, RESULTS_DIR


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load experiment configuration from YAML file."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return ExperimentConfig.from_yaml(data)


def expand_matrix(config: ExperimentConfig) -> list[dict]:
    """
    Expand experiment matrix into all combinations.

    Returns:
        List of dicts with keys: model, temperature, evaluator_prompt, run_number
    """
    combinations = []

    for model, temp, prompt in itertools.product(
        config.models,
        config.temperatures,
        config.evaluator_prompts,
    ):
        for run_num in range(config.runs_per_combination):
            combinations.append({
                "model": model,
                "temperature": temp,
                "evaluator_prompt": prompt,
                "run_number": run_num + 1,
            })

    return combinations


def resolve_arguments(config: ExperimentConfig, arguments_dir: Path = ARGUMENTS_DIR) -> list[Argument]:
    """
    Resolve argument IDs to actual Argument objects.

    Args:
        config: Experiment config with argument_ids
        arguments_dir: Arguments directory

    Returns:
        List of Arguments
    """
    arguments = []

    for arg_id in config.argument_ids:
        arg = load_argument(arg_id, arguments_dir)
        if arg:
            arguments.append(arg)
        else:
            print(f"  Warning: Argument {arg_id} not found, skipping")

    return arguments


def _evaluate_single(
    model_id: str,
    actual_model_id: str,
    argument: Argument,
    temperature: float,
    prompt_name: str,
    run_id: str,
    config_dir: Path,
    app_config,
) -> tuple[str, EvaluationResult | None, str | None]:
    """
    Evaluate a single argument with a single model. Used for parallel execution.

    Returns:
        Tuple of (run_id, evaluation_result, error_message)
    """
    try:
        provider = get_provider_for_model(model_id, app_config)
        evaluation = evaluate_argument(
            provider=provider,
            argument=argument,
            model=actual_model_id,
            temperature=temperature,
            prompt_name=prompt_name,
            run_id=run_id,
            config_dir=config_dir,
        )
        return (run_id, evaluation, None)
    except Exception as e:
        return (run_id, None, str(e))


def run_experiment_parallel(
    config: ExperimentConfig,
    arguments: list[Argument] | None = None,
    arguments_dir: Path = ARGUMENTS_DIR,
    results_dir: Path = RESULTS_DIR,
    config_dir: Path = CONFIG_DIR,
    max_workers: int | None = None,
) -> ExperimentResults:
    """
    Run experiment with parallel execution across models.

    For each argument, evaluations across different models run in parallel.
    This is safe because each model uses a different API endpoint.

    Args:
        config: Experiment configuration
        arguments: List of arguments to evaluate
        arguments_dir: Arguments directory
        results_dir: Results directory
        config_dir: Config directory
        max_workers: Max parallel workers (default: number of models)

    Returns:
        ExperimentResults with all evaluations
    """
    # Load arguments if not provided
    if arguments is None:
        arguments = resolve_arguments(config, arguments_dir)

    if not arguments:
        raise ValueError("No arguments to evaluate")

    # Create results container
    results = ExperimentResults.create(config)

    # Expand the test matrix
    matrix = expand_matrix(config)
    total_evaluations = len(matrix) * len(arguments)

    # Default workers = number of unique models (parallel across providers)
    if max_workers is None:
        max_workers = len(config.models)

    print(f"\nExperiment: {config.name} (PARALLEL MODE)")
    print(f"  Models: {config.models}")
    print(f"  Temperatures: {config.temperatures}")
    print(f"  Prompts: {config.evaluator_prompts}")
    print(f"  Arguments: {len(arguments)}")
    print(f"  Runs per combination: {config.runs_per_combination}")
    print(f"  Total evaluations: {total_evaluations}")
    print(f"  Parallel workers: {max_workers}")
    print()

    # Load global config for model lookups
    app_config = load_config(config_dir)

    # Pre-resolve model IDs
    model_id_map = {}
    for model_id in config.models:
        model_config = app_config.models.get(model_id)
        if model_config:
            model_id_map[model_id] = model_config.model_id
        else:
            model_id_map[model_id] = model_id

    completed = 0
    results_lock = Lock()

    # Process each argument, running all model evaluations in parallel
    for arg_idx, argument in enumerate(arguments):
        print(f"\n[Argument {arg_idx + 1}/{len(arguments)}: {argument.id}]")

        # Build tasks for this argument (all model/temp/prompt combos)
        tasks = []
        for combo in matrix:
            model_id = combo["model"]
            temperature = combo["temperature"]
            prompt_name = combo["evaluator_prompt"]
            run_number = combo["run_number"]
            run_id = f"{model_id}_{temperature}_{prompt_name}_run{run_number}"

            tasks.append({
                "model_id": model_id,
                "actual_model_id": model_id_map.get(model_id, model_id),
                "argument": argument,
                "temperature": temperature,
                "prompt_name": prompt_name,
                "run_id": run_id,
            })

        # Run evaluations in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _evaluate_single,
                    task["model_id"],
                    task["actual_model_id"],
                    task["argument"],
                    task["temperature"],
                    task["prompt_name"],
                    task["run_id"],
                    config_dir,
                    app_config,
                ): task
                for task in tasks
            }

            for future in as_completed(futures):
                task = futures[future]
                run_id, evaluation, error = future.result()

                completed += 1

                if evaluation:
                    with results_lock:
                        results.add_evaluation(evaluation)

                    delta = evaluation.score_delta
                    print(f"  [{completed}/{total_evaluations}] {task['model_id']}: "
                          f"Score {evaluation.score} (expected: {argument.expected_score}, delta: {delta:+d})")
                else:
                    print(f"  [{completed}/{total_evaluations}] {task['model_id']}: Error - {error}")

        # Save after each argument (incremental)
        save_results(results, results_dir)

    # Mark complete and final save
    results.mark_complete()
    save_results(results, results_dir)

    print(f"\nExperiment complete! Results saved to: {results_dir / results.experiment_id}")

    return results


def run_experiment(
    config: ExperimentConfig,
    arguments: list[Argument] | None = None,
    arguments_dir: Path = ARGUMENTS_DIR,
    results_dir: Path = RESULTS_DIR,
    config_dir: Path = CONFIG_DIR,
    incremental_save: bool = True,
    parallel: bool = False,
    max_workers: int | None = None,
) -> ExperimentResults:
    """
    Run a complete experiment.

    Args:
        config: Experiment configuration
        arguments: List of arguments to evaluate (loads from config.argument_ids if None)
        arguments_dir: Arguments directory
        results_dir: Results directory
        config_dir: Config directory
        incremental_save: Save after each evaluation (for crash recovery)
        parallel: Run evaluations in parallel across models
        max_workers: Max parallel workers (only used if parallel=True)

    Returns:
        ExperimentResults with all evaluations
    """
    if parallel:
        return run_experiment_parallel(
            config=config,
            arguments=arguments,
            arguments_dir=arguments_dir,
            results_dir=results_dir,
            config_dir=config_dir,
            max_workers=max_workers,
        )

    # Load arguments if not provided
    if arguments is None:
        arguments = resolve_arguments(config, arguments_dir)

    if not arguments:
        raise ValueError("No arguments to evaluate")

    # Create results container
    results = ExperimentResults.create(config)

    # Expand the test matrix
    matrix = expand_matrix(config)
    total_evaluations = len(matrix) * len(arguments)

    print(f"\nExperiment: {config.name}")
    print(f"  Models: {config.models}")
    print(f"  Temperatures: {config.temperatures}")
    print(f"  Prompts: {config.evaluator_prompts}")
    print(f"  Arguments: {len(arguments)}")
    print(f"  Runs per combination: {config.runs_per_combination}")
    print(f"  Total evaluations: {total_evaluations}")
    print()

    eval_count = 0

    # Load global config for model lookups
    app_config = load_config(config_dir)

    for combo in matrix:
        model_id = combo["model"]
        temperature = combo["temperature"]
        prompt_name = combo["evaluator_prompt"]
        run_number = combo["run_number"]

        run_id = f"{model_id}_{temperature}_{prompt_name}_run{run_number}"

        print(f"[{run_id}]")

        # Get the provider for this model
        try:
            provider = get_provider_for_model(model_id, app_config)
        except ValueError as e:
            print(f"  Error: {e}, skipping")
            continue

        # Get the actual model ID from config
        model_config = app_config.models.get(model_id)
        actual_model_id = model_config.model_id if model_config else model_id

        for argument in arguments:
            eval_count += 1
            print(f"  [{eval_count}/{total_evaluations}] Evaluating {argument.id}...", end=" ")

            try:
                evaluation = evaluate_argument(
                    provider=provider,
                    argument=argument,
                    model=actual_model_id,
                    temperature=temperature,
                    prompt_name=prompt_name,
                    run_id=run_id,
                    config_dir=config_dir,
                )

                print(f"Score: {evaluation.score} (expected: {argument.expected_score}, delta: {evaluation.score_delta:+d})")

                if incremental_save:
                    append_evaluation(results, evaluation, results_dir)
                else:
                    results.add_evaluation(evaluation)

            except Exception as e:
                print(f"Error: {e}")

    # Mark complete and final save
    results.mark_complete()
    save_results(results, results_dir)

    print(f"\nExperiment complete! Results saved to: {results_dir / results.experiment_id}")

    return results


def run_experiment_from_file(
    experiment_path: Path,
    arguments_dir: Path = ARGUMENTS_DIR,
    results_dir: Path = RESULTS_DIR,
    config_dir: Path = CONFIG_DIR,
    parallel: bool = False,
) -> ExperimentResults:
    """
    Run an experiment from a YAML file.

    Args:
        experiment_path: Path to experiment YAML file
        arguments_dir: Arguments directory
        results_dir: Results directory
        config_dir: Config directory
        parallel: Run evaluations in parallel

    Returns:
        ExperimentResults
    """
    config = load_experiment_config(experiment_path)
    return run_experiment(
        config=config,
        arguments_dir=arguments_dir,
        results_dir=results_dir,
        config_dir=config_dir,
        parallel=parallel,
    )

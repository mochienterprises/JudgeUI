"""Blind argument evaluation with fault detection."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import json
from pathlib import Path
from threading import Lock
from typing import Any

from .config import load_prompt, CONFIG_DIR
from .models import Argument, EvaluationResult
from .providers import LLMProvider


@dataclass
class JudgeResult:
    """Result of a single judge evaluating an argument."""
    judge_model: str
    evaluation: EvaluationResult | None
    error: str | None

    def to_dict(self) -> dict:
        return {
            "judge_model": self.judge_model,
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
            "error": self.error,
        }


@dataclass
class ComparisonMatrix:
    """Matrix of evaluation results: arguments x judges."""
    results: dict[str, dict[str, JudgeResult]]  # {arg_id: {judge_id: JudgeResult}}
    arguments: dict[str, Argument]

    def to_dict(self) -> dict:
        return {
            "results": {
                arg_id: {
                    judge_id: result.to_dict()
                    for judge_id, result in judge_results.items()
                }
                for arg_id, judge_results in self.results.items()
            },
            "arguments": {
                arg_id: arg.to_dict()
                for arg_id, arg in self.arguments.items()
            }
        }

    def get_summary_stats(self) -> dict[str, Any]:
        """Compute summary statistics for the comparison matrix."""
        by_judge: dict[str, dict[str, Any]] = {}
        by_argument: dict[str, dict[str, Any]] = {}
        all_scores: list[int] = []
        successes = 0
        failures = 0

        for arg_id, judge_results in self.results.items():
            arg_scores: list[int] = []

            for judge_id, result in judge_results.items():
                if result.error:
                    failures += 1
                    continue

                successes += 1
                if result.evaluation:
                    score = result.evaluation.score
                    all_scores.append(score)
                    arg_scores.append(score)

                    # Track by judge
                    if judge_id not in by_judge:
                        by_judge[judge_id] = {"scores": [], "count": 0}
                    by_judge[judge_id]["scores"].append(score)
                    by_judge[judge_id]["count"] += 1

            # Compute stats for this argument
            if arg_scores:
                by_argument[arg_id] = {
                    "avg_score": round(sum(arg_scores) / len(arg_scores), 1),
                    "min_score": min(arg_scores),
                    "max_score": max(arg_scores),
                    "count": len(arg_scores),
                }

        # Compute averages for judges
        for judge_id in by_judge:
            scores = by_judge[judge_id]["scores"]
            by_judge[judge_id]["avg_score"] = round(sum(scores) / len(scores), 1) if scores else 0
            del by_judge[judge_id]["scores"]  # Remove raw scores from output

        # Compute overall stats
        overall = {
            "avg_score": round(sum(all_scores) / len(all_scores), 1) if all_scores else 0,
            "successes": successes,
            "failures": failures,
        }

        return {
            "by_judge": by_judge,
            "by_argument": by_argument,
            "overall": overall,
        }


def parse_evaluation_response(response_text: str) -> dict:
    """Parse JSON response from evaluator, handling markdown code blocks."""
    text = response_text.strip()

    # Handle markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (``` markers)
        if len(lines) > 2:
            text = "\n".join(lines[1:-1])
        # Remove json language identifier if present
        if text.startswith("json"):
            text = text[4:].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback for malformed responses
        return {
            "score": 50,
            "detected_faults": [],
            "reasoning": "Error parsing evaluation response",
        }


def evaluate_argument(
    provider: LLMProvider,
    argument: Argument,
    model: str | None = None,
    temperature: float = 1.0,
    prompt_name: str = "default",
    system_prompt: str | None = None,
    run_id: str = "",
    config_dir: Path = CONFIG_DIR,
) -> EvaluationResult:
    """
    Evaluate an argument blind (without topic context).

    Args:
        provider: LLM provider to use
        argument: Argument to evaluate
        model: Model ID (provider default if None)
        temperature: Sampling temperature
        prompt_name: Name of evaluator prompt to use
        system_prompt: Optional override system prompt
        run_id: ID for this evaluation run
        config_dir: Config directory path

    Returns:
        EvaluationResult with scores and detected faults
    """
    # Load and format the evaluation prompt
    prompt_template = load_prompt(prompt_name, "evaluators", config_dir)
    prompt = prompt_template.format(argument=argument.text)

    # Call the evaluator
    response = provider.call(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        model=model,
        max_tokens=1000,
    )

    # Parse the response
    parsed = parse_evaluation_response(response.text)

    # Create the evaluation result
    result = EvaluationResult(
        argument_id=argument.id,
        model=response.model,
        temperature=temperature,
        system_prompt=prompt_name,
        score=parsed.get("score", 50),
        detected_faults=parsed.get("detected_faults", []),
        reasoning=parsed.get("reasoning", ""),
        run_id=run_id,
    )

    # Compute ground truth metrics
    result.compute_metrics(argument)

    return result


def evaluate_arguments(
    provider: LLMProvider,
    arguments: list[Argument],
    model: str | None = None,
    temperature: float = 1.0,
    prompt_name: str = "default",
    run_id: str = "",
    config_dir: Path = CONFIG_DIR,
) -> list[EvaluationResult]:
    """
    Evaluate multiple arguments.

    Args:
        provider: LLM provider
        arguments: List of arguments to evaluate
        model: Model ID
        temperature: Sampling temperature
        prompt_name: Evaluator prompt name
        run_id: ID for this evaluation run
        config_dir: Config directory

    Returns:
        List of EvaluationResults
    """
    results = []

    for i, argument in enumerate(arguments):
        print(f"  Evaluating argument {i + 1}/{len(arguments)} (ID: {argument.id})...")
        result = evaluate_argument(
            provider=provider,
            argument=argument,
            model=model,
            temperature=temperature,
            prompt_name=prompt_name,
            run_id=run_id,
            config_dir=config_dir,
        )
        results.append(result)

    return results


def _evaluate_single_judge(
    judge_id: str,
    actual_model_id: str,
    provider: LLMProvider,
    argument: Argument,
    temperature: float,
    prompt_name: str,
    run_id: str,
    config_dir: Path,
) -> JudgeResult:
    """
    Worker function for parallel evaluation with a single judge.

    Args:
        judge_id: Display model ID for results
        actual_model_id: Actual model ID to pass to provider
        provider: LLM provider instance
        argument: Argument to evaluate
        temperature: Sampling temperature
        prompt_name: Name of evaluator prompt
        run_id: ID for this evaluation run
        config_dir: Config directory path

    Returns:
        JudgeResult with evaluation or error
    """
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
        return JudgeResult(judge_model=judge_id, evaluation=evaluation, error=None)
    except Exception as e:
        return JudgeResult(judge_model=judge_id, evaluation=None, error=str(e))


def evaluate_with_multiple_judges(
    judges: dict[str, tuple[LLMProvider, str]],  # {judge_id: (provider, actual_model_id)}
    argument: Argument,
    temperature: float = 1.0,
    prompt_name: str = "default",
    run_id: str = "",
    config_dir: Path = CONFIG_DIR,
    max_workers: int | None = None,
) -> dict[str, JudgeResult]:
    """
    Evaluate a single argument with multiple judges in parallel.

    Args:
        judges: Dict mapping judge_id to (provider, actual_model_id) tuples
        argument: Argument to evaluate
        temperature: Sampling temperature
        prompt_name: Name of evaluator prompt
        run_id: ID for this evaluation run
        config_dir: Config directory path
        max_workers: Maximum number of parallel workers

    Returns:
        Dict mapping judge_id to JudgeResult
    """
    max_workers = max_workers or len(judges)

    results: dict[str, JudgeResult] = {}
    results_lock = Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _evaluate_single_judge,
                judge_id,
                actual_model_id,
                provider,
                argument,
                temperature,
                prompt_name,
                run_id,
                config_dir,
            ): judge_id
            for judge_id, (provider, actual_model_id) in judges.items()
        }

        for future in as_completed(futures):
            judge_id = futures[future]
            try:
                result = future.result()
                with results_lock:
                    results[judge_id] = result
            except Exception as e:
                with results_lock:
                    results[judge_id] = JudgeResult(
                        judge_model=judge_id, evaluation=None, error=str(e)
                    )

    return results


def evaluate_arguments_comparison(
    judges: dict[str, tuple[LLMProvider, str]],
    arguments: dict[str, Argument],  # {arg_id: Argument}
    temperature: float = 1.0,
    prompt_name: str = "default",
    run_id: str = "",
    config_dir: Path = CONFIG_DIR,
    max_workers: int | None = None,
) -> ComparisonMatrix:
    """
    Evaluate multiple arguments with multiple judges.

    Args:
        judges: Dict mapping judge_id to (provider, actual_model_id) tuples
        arguments: Dict mapping arg_id to Argument
        temperature: Sampling temperature
        prompt_name: Name of evaluator prompt
        run_id: ID for this evaluation run
        config_dir: Config directory path
        max_workers: Maximum number of parallel workers

    Returns:
        ComparisonMatrix with all evaluation results
    """
    # Calculate total tasks for worker pool
    total_tasks = len(judges) * len(arguments)
    max_workers = max_workers or min(total_tasks, 10)  # Cap at 10 concurrent

    results: dict[str, dict[str, JudgeResult]] = {arg_id: {} for arg_id in arguments}
    results_lock = Lock()

    # Build list of all (arg_id, judge_id) tasks
    tasks = [
        (arg_id, judge_id, actual_model_id, provider, arguments[arg_id])
        for arg_id in arguments
        for judge_id, (provider, actual_model_id) in judges.items()
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _evaluate_single_judge,
                judge_id,
                actual_model_id,
                provider,
                argument,
                temperature,
                prompt_name,
                run_id,
                config_dir,
            ): (arg_id, judge_id)
            for arg_id, judge_id, actual_model_id, provider, argument in tasks
        }

        for future in as_completed(futures):
            arg_id, judge_id = futures[future]
            try:
                result = future.result()
                with results_lock:
                    results[arg_id][judge_id] = result
            except Exception as e:
                with results_lock:
                    results[arg_id][judge_id] = JudgeResult(
                        judge_model=judge_id, evaluation=None, error=str(e)
                    )

    return ComparisonMatrix(results=results, arguments=arguments)


def cross_evaluate_arguments(
    judges: dict[str, tuple[LLMProvider, str]],
    arguments: dict[str, Argument],  # {generator_model: Argument}
    exclude_self_evaluation: bool = True,
    temperature: float = 1.0,
    prompt_name: str = "default",
    run_id: str = "",
    config_dir: Path = CONFIG_DIR,
    max_workers: int | None = None,
) -> ComparisonMatrix:
    """
    Cross-evaluate: each judge evaluates each argument, optionally excluding self-evaluation.

    Args:
        judges: Dict mapping judge_id to (provider, actual_model_id) tuples
        arguments: Dict mapping generator_model to Argument (the model that generated it)
        exclude_self_evaluation: If True, skip evaluations where generator == judge
        temperature: Sampling temperature
        prompt_name: Name of evaluator prompt
        run_id: ID for this evaluation run
        config_dir: Config directory path
        max_workers: Maximum number of parallel workers

    Returns:
        ComparisonMatrix with cross-evaluation results (N/A for self-evaluations if excluded)
    """
    results: dict[str, dict[str, JudgeResult]] = {arg_id: {} for arg_id in arguments}
    results_lock = Lock()

    # Build list of tasks, optionally excluding self-evaluation
    tasks = []
    for generator_model, argument in arguments.items():
        for judge_id, (provider, actual_model_id) in judges.items():
            if exclude_self_evaluation and generator_model == judge_id:
                # Mark as skipped (self-evaluation)
                results[generator_model][judge_id] = JudgeResult(
                    judge_model=judge_id,
                    evaluation=None,
                    error="Self-evaluation excluded"
                )
            else:
                tasks.append((generator_model, judge_id, actual_model_id, provider, argument))

    if not tasks:
        return ComparisonMatrix(results=results, arguments=arguments)

    max_workers = max_workers or min(len(tasks), 10)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _evaluate_single_judge,
                judge_id,
                actual_model_id,
                provider,
                argument,
                temperature,
                prompt_name,
                run_id,
                config_dir,
            ): (generator_model, judge_id)
            for generator_model, judge_id, actual_model_id, provider, argument in tasks
        }

        for future in as_completed(futures):
            generator_model, judge_id = futures[future]
            try:
                result = future.result()
                with results_lock:
                    results[generator_model][judge_id] = result
            except Exception as e:
                with results_lock:
                    results[generator_model][judge_id] = JudgeResult(
                        judge_model=judge_id, evaluation=None, error=str(e)
                    )

    return ComparisonMatrix(results=results, arguments=arguments)

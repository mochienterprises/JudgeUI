#!/usr/bin/env python3
"""
CMV (ChangeMyView) Winning Arguments Experiment

Tests AI judges on their ability to distinguish persuasive (winning)
arguments from non-persuasive (losing) arguments using real data from
Reddit's r/ChangeMyView subreddit.

The key metric is: Can the AI judge reliably score winning arguments
higher than losing arguments from the same conversation?
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv

load_dotenv(override=True)

from src.config import load_config, get_provider_for_model, CONFIG_DIR
from src.convokit_loader import WinningArgumentsLoader, ArgumentPair
from src.models import Argument


def load_evaluator_prompt(prompt_name: str, config_dir: Path = CONFIG_DIR) -> str:
    """Load evaluator system prompt."""
    prompt_path = config_dir / "prompts" / "evaluators" / f"{prompt_name}.txt"
    if not prompt_path.exists():
        raise ValueError(f"Prompt not found: {prompt_path}")
    return prompt_path.read_text()


def evaluate_argument(
    provider,
    argument: Argument,
    model: str,
    temperature: float,
    system_prompt: str,
) -> dict:
    """
    Evaluate a single argument and return score.

    Returns:
        Dict with score, reasoning, raw_response
    """
    user_prompt = f"""Please evaluate the following argument:

TOPIC: {argument.topic}

ARGUMENT:
{argument.text}

Provide your evaluation with a score from 0-100 and brief reasoning."""

    response = provider.call(
        prompt=user_prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=500,
        model=model,
    )

    # Parse score from response
    text = response.text
    score = None

    # Try to find score in various formats
    import re
    patterns = [
        r"(?:score|rating)[:\s]*(\d+)",
        r"(\d+)(?:\s*(?:out of|/)\s*100)",
        r"^\s*(\d+)\s*$",
    ]

    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            score = int(match.group(1))
            if score > 100:
                score = None
            else:
                break

    # If no score found, try to extract first number
    if score is None:
        numbers = re.findall(r"\b(\d+)\b", text)
        for num in numbers:
            n = int(num)
            if 0 <= n <= 100:
                score = n
                break

    return {
        "score": score,
        "reasoning": text[:500],
        "raw_response": text,
    }


def evaluate_pair(
    pair: ArgumentPair,
    provider,
    model: str,
    actual_model_id: str,
    temperature: float,
    system_prompt: str,
) -> dict:
    """
    Evaluate both arguments in a pair and determine if judge correctly
    ranked the winning argument higher.
    """
    # Evaluate both arguments
    win_result = evaluate_argument(
        provider=provider,
        argument=pair.successful,
        model=actual_model_id,
        temperature=temperature,
        system_prompt=system_prompt,
    )

    lose_result = evaluate_argument(
        provider=provider,
        argument=pair.unsuccessful,
        model=actual_model_id,
        temperature=temperature,
        system_prompt=system_prompt,
    )

    win_score = win_result["score"]
    lose_score = lose_result["score"]

    # Determine if judge correctly ranked
    if win_score is not None and lose_score is not None:
        correct = win_score > lose_score
        tie = win_score == lose_score
        margin = win_score - lose_score
    else:
        correct = None
        tie = None
        margin = None

    return {
        "pair_id": pair.pair_id,
        "conversation_id": pair.conversation_id,
        "topic": pair.topic,
        "model": model,
        "temperature": temperature,
        "winning_arg": {
            "id": pair.successful.id,
            "score": win_score,
            "text_length": len(pair.successful.text),
            "reasoning": win_result["reasoning"],
        },
        "losing_arg": {
            "id": pair.unsuccessful.id,
            "score": lose_score,
            "text_length": len(pair.unsuccessful.text),
            "reasoning": lose_result["reasoning"],
        },
        "correct_ranking": correct,
        "is_tie": tie,
        "score_margin": margin,
    }


def run_cmv_experiment(
    models: list[str],
    num_pairs: int = 20,
    temperature: float = 0.0,
    prompt_name: str = "default",
    seed: int = 42,
    max_workers: int | None = None,
    min_text_length: int = 200,
    max_text_length: int = 1500,
) -> dict:
    """
    Run CMV experiment across models.

    Args:
        models: List of model IDs to test
        num_pairs: Number of argument pairs to evaluate
        temperature: Model temperature
        prompt_name: Name of evaluator prompt to use
        seed: Random seed for reproducibility
        max_workers: Max parallel workers
        min_text_length: Minimum argument text length
        max_text_length: Maximum argument text length

    Returns:
        Dict with all results and summary
    """
    # Load data
    print("Loading ChangeMyView corpus...")
    loader = WinningArgumentsLoader()
    pairs = loader.sample_pairs(
        n=num_pairs,
        seed=seed,
        min_text_length=min_text_length,
        max_text_length=max_text_length,
    )
    print(f"  Loaded {len(pairs)} argument pairs")

    # Load config
    app_config = load_config(CONFIG_DIR)
    system_prompt = load_evaluator_prompt(prompt_name, CONFIG_DIR)

    if max_workers is None:
        max_workers = len(models)

    print(f"\nCMV Winning Arguments Experiment")
    print("=" * 50)
    print(f"Models: {models}")
    print(f"Pairs: {len(pairs)}")
    print(f"Temperature: {temperature}")
    print(f"Prompt: {prompt_name}")
    print()

    # Pre-resolve model IDs
    model_info = {}
    for model_id in models:
        try:
            provider = get_provider_for_model(model_id, app_config)
            model_config = app_config.models.get(model_id)
            actual_model_id = model_config.model_id if model_config else model_id
            model_info[model_id] = {
                "provider": provider,
                "actual_model_id": actual_model_id,
            }
        except Exception as e:
            print(f"  Error initializing {model_id}: {e}")

    results = {}
    results_lock = Lock()

    def evaluate_model(model_id: str) -> tuple[str, list[dict], str | None]:
        info = model_info.get(model_id)
        if not info:
            return (model_id, [], "Failed to initialize provider")

        model_results = []
        for i, pair in enumerate(pairs):
            try:
                result = evaluate_pair(
                    pair=pair,
                    provider=info["provider"],
                    model=model_id,
                    actual_model_id=info["actual_model_id"],
                    temperature=temperature,
                    system_prompt=system_prompt,
                )
                model_results.append(result)
            except Exception as e:
                model_results.append({
                    "pair_id": pair.pair_id,
                    "error": str(e),
                })

        return (model_id, model_results, None)

    # Run evaluations in parallel across models
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(evaluate_model, model_id): model_id
            for model_id in models
            if model_id in model_info
        }

        for future in as_completed(futures):
            model_id, model_results, error = future.result()

            if error:
                print(f"  {model_id}: Error - {error}")
                continue

            with results_lock:
                results[model_id] = model_results

            # Calculate accuracy
            valid = [r for r in model_results if r.get("correct_ranking") is not None]
            correct = sum(1 for r in valid if r["correct_ranking"])
            ties = sum(1 for r in valid if r.get("is_tie"))
            accuracy = correct / len(valid) if valid else 0

            avg_margin = sum(r["score_margin"] for r in valid if r["score_margin"]) / len(valid) if valid else 0

            print(f"  {model_id}:")
            print(f"    Accuracy: {correct}/{len(valid)} ({accuracy:.1%})")
            print(f"    Ties: {ties}")
            print(f"    Avg margin: {avg_margin:+.1f}")

    # Build summary
    summary = {
        "experiment": "CMV Winning Arguments",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_pairs": num_pairs,
            "temperature": temperature,
            "prompt": prompt_name,
            "seed": seed,
        },
        "model_performance": {},
    }

    for model_id, model_results in results.items():
        valid = [r for r in model_results if r.get("correct_ranking") is not None]
        correct = sum(1 for r in valid if r["correct_ranking"])
        ties = sum(1 for r in valid if r.get("is_tie"))

        summary["model_performance"][model_id] = {
            "total_pairs": len(model_results),
            "valid_pairs": len(valid),
            "correct": correct,
            "ties": ties,
            "accuracy": correct / len(valid) if valid else 0,
            "avg_win_score": sum(r["winning_arg"]["score"] for r in valid if r["winning_arg"]["score"]) / len(valid) if valid else 0,
            "avg_lose_score": sum(r["losing_arg"]["score"] for r in valid if r["losing_arg"]["score"]) / len(valid) if valid else 0,
        }

    return {
        "summary": summary,
        "results": results,
        "pairs_info": [{"pair_id": p.pair_id, "topic": p.topic} for p in pairs],
    }


def save_results(results: dict, output_dir: Path) -> Path:
    """Save results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cmv_experiment_{timestamp}.json"
    output_path = output_dir / filename

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return output_path


def print_summary(results: dict):
    """Print formatted summary."""
    summary = results["summary"]

    print("\n" + "=" * 60)
    print("CMV EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)

    # Sort by accuracy
    sorted_models = sorted(
        summary["model_performance"].items(),
        key=lambda x: x[1]["accuracy"],
        reverse=True,
    )

    print("\nModel Rankings (by accuracy in identifying winning arguments):")
    print("-" * 50)

    for rank, (model_id, perf) in enumerate(sorted_models, 1):
        print(f"  {rank}. {model_id}")
        print(f"     Accuracy: {perf['correct']}/{perf['valid_pairs']} ({perf['accuracy']:.1%})")
        print(f"     Avg scores: Win={perf['avg_win_score']:.0f}, Lose={perf['avg_lose_score']:.0f}")
        print(f"     Ties: {perf['ties']}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Test AI judges on CMV winning vs losing arguments"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-5.2", "grok-3", "gemini-2.0-flash"],
        help="Model IDs to test"
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=20,
        help="Number of argument pairs to evaluate (default: 20)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for model responses (default: 0.0)"
    )
    parser.add_argument(
        "--prompt",
        default="default",
        help="Evaluator prompt to use (default: default)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for pair selection (default: 42)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/cmv"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        help="Number of parallel workers"
    )

    args = parser.parse_args()

    results = run_cmv_experiment(
        models=args.models,
        num_pairs=args.pairs,
        temperature=args.temperature,
        prompt_name=args.prompt,
        seed=args.seed,
        max_workers=args.parallel,
    )

    output_path = save_results(results, args.output_dir)
    print(f"\nResults saved to: {output_path}")

    print_summary(results)


if __name__ == "__main__":
    main()

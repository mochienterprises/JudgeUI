"""Blind argument evaluation with fault detection."""

import json
from pathlib import Path

from .config import load_prompt, CONFIG_DIR
from .models import Argument, EvaluationResult
from .providers import LLMProvider


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

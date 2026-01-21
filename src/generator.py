"""Argument generation with fault injection."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Literal

from .config import load_faults, load_prompt, CONFIG_DIR, FaultConfig
from .models import Argument
from .providers import LLMProvider


@dataclass
class GenerationResult:
    """Result of generating an argument with a specific model."""
    model_id: str
    argument: Argument | None
    error: str | None

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "argument": self.argument.to_dict() if self.argument else None,
            "error": self.error,
        }


def build_fault_instructions(faults: list[str], fault_configs: dict[str, FaultConfig]) -> str:
    """Build generation instructions for specified faults."""
    if not faults:
        return """QUALITY LEVEL: HIGH
Write a well-reasoned, logical argument with:
- Clear evidence and citations where appropriate
- Sound logical structure
- Acknowledgment of nuance and counterarguments
- No logical fallacies or manipulation tactics"""

    instructions = ["QUALITY LEVEL: FLAWED", "Include the following specific flaws in your argument:", ""]

    for fault_id in faults:
        if fault_id in fault_configs:
            fault = fault_configs[fault_id]
            instructions.append(f"- {fault_id.upper()}: {fault.generation_hint}")
        else:
            instructions.append(f"- {fault_id.upper()}: Include this flaw")

    return "\n".join(instructions)


def calculate_expected_score(faults: list[str], fault_configs: dict[str, FaultConfig]) -> int:
    """Calculate expected score based on fault severities."""
    if not faults:
        return 100

    total_penalty = 0
    for fault_id in faults:
        if fault_id in fault_configs:
            total_penalty += abs(fault_configs[fault_id].severity)
        else:
            total_penalty += 10  # Default penalty for unknown faults

    return max(0, 100 - total_penalty)


def generate_argument(
    provider: LLMProvider,
    topic: str,
    stance: Literal["for", "against"],
    faults: list[str] | None = None,
    model: str | None = None,
    prompt_name: str = "default",
    config_dir: Path = CONFIG_DIR,
) -> Argument:
    """
    Generate an argument with optional fault injection.

    Args:
        provider: LLM provider to use for generation
        topic: Debate topic
        stance: "for" or "against"
        faults: List of fault IDs to inject (empty = clean argument)
        model: Model ID to use (provider default if None)
        prompt_name: Name of generator prompt to use
        config_dir: Config directory path

    Returns:
        Generated Argument with metadata
    """
    faults = faults or []
    fault_configs = load_faults(config_dir)

    # Load and format the generation prompt
    prompt_template = load_prompt(prompt_name, "generators", config_dir)
    fault_instructions = build_fault_instructions(faults, fault_configs)

    prompt = prompt_template.format(
        topic=topic,
        stance=stance.upper(),
        fault_instructions=fault_instructions,
    )

    # Generate the argument
    response = provider.call(
        prompt=prompt,
        model=model,
        max_tokens=2000,
    )

    # Calculate expected score
    expected_score = calculate_expected_score(faults, fault_configs)

    return Argument.create(
        topic=topic,
        stance=stance,
        text=response.text,
        injected_faults=faults,
        expected_score=expected_score,
        source="generated",
        generated_by=response.model,
    )


def generate_argument_set(
    provider: LLMProvider,
    topic: str,
    fault_combinations: list[list[str]] | None = None,
    model: str | None = None,
    config_dir: Path = CONFIG_DIR,
) -> list[Argument]:
    """
    Generate a set of arguments for a topic with different fault levels.

    Args:
        provider: LLM provider
        topic: Debate topic
        fault_combinations: List of fault lists to generate (default: clean, 2 faults, 4 faults)
        model: Model ID
        config_dir: Config directory

    Returns:
        List of generated Arguments
    """
    if fault_combinations is None:
        # Default: clean, moderate faults, heavy faults for both stances
        # Using only hard faults (objectively detrimental)
        fault_combinations = [
            [],  # Clean argument
            ["hasty_generalization", "cherry_picking"],  # Moderate: 2 faults (-22)
            ["strawman", "ad_hominem", "no_evidence", "circular_reasoning"],  # Heavy: 4 faults (-60)
        ]

    arguments = []

    for stance in ["for", "against"]:
        for faults in fault_combinations:
            arg = generate_argument(
                provider=provider,
                topic=topic,
                stance=stance,
                faults=faults,
                model=model,
                config_dir=config_dir,
            )
            arguments.append(arg)
            print(f"  Generated {stance.upper()} argument with {len(faults)} faults")

    return arguments


def _generate_single(
    model_id: str,
    actual_model_id: str,
    provider: LLMProvider,
    topic: str,
    stance: Literal["for", "against"],
    faults: list[str],
    prompt_name: str,
    config_dir: Path,
) -> GenerationResult:
    """
    Worker function for parallel generation.

    Args:
        model_id: Display model ID for results
        actual_model_id: Actual model ID to pass to provider
        provider: LLM provider instance
        topic: Debate topic
        stance: "for" or "against"
        faults: List of fault IDs to inject
        prompt_name: Name of generator prompt
        config_dir: Config directory path

    Returns:
        GenerationResult with argument or error
    """
    try:
        argument = generate_argument(
            provider=provider,
            topic=topic,
            stance=stance,
            faults=faults,
            model=actual_model_id,
            prompt_name=prompt_name,
            config_dir=config_dir,
        )
        return GenerationResult(model_id=model_id, argument=argument, error=None)
    except Exception as e:
        return GenerationResult(model_id=model_id, argument=None, error=str(e))


def generate_argument_comparison(
    models: dict[str, tuple[LLMProvider, str]],  # {model_id: (provider, actual_model_id)}
    topic: str,
    stance: Literal["for", "against"],
    faults: list[str] | None = None,
    prompt_name: str = "default",
    config_dir: Path = CONFIG_DIR,
    max_workers: int | None = None,
) -> dict[str, GenerationResult]:
    """
    Generate the same argument with multiple models in parallel.

    Args:
        models: Dict mapping model_id to (provider, actual_model_id) tuples
        topic: Debate topic
        stance: "for" or "against"
        faults: List of fault IDs to inject (empty = clean argument)
        prompt_name: Name of generator prompt to use
        config_dir: Config directory path
        max_workers: Maximum number of parallel workers (defaults to number of models)

    Returns:
        Dict mapping model_id to GenerationResult
    """
    faults = faults or []
    max_workers = max_workers or len(models)

    results: dict[str, GenerationResult] = {}
    results_lock = Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _generate_single,
                model_id,
                actual_model_id,
                provider,
                topic,
                stance,
                faults,
                prompt_name,
                config_dir,
            ): model_id
            for model_id, (provider, actual_model_id) in models.items()
        }

        for future in as_completed(futures):
            model_id = futures[future]
            try:
                result = future.result()
                with results_lock:
                    results[model_id] = result
            except Exception as e:
                with results_lock:
                    results[model_id] = GenerationResult(
                        model_id=model_id, argument=None, error=str(e)
                    )

    return results

"""JSON-based storage for arguments and experiment results."""

import json
from pathlib import Path
from typing import Iterator

from .models import Argument, ExperimentResults


# Default directories
ARGUMENTS_DIR = Path(__file__).parent.parent / "arguments"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def ensure_dir(path: Path) -> None:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)


# --- Argument Storage ---

def save_argument(argument: Argument, arguments_dir: Path = ARGUMENTS_DIR) -> Path:
    """
    Save an argument to the appropriate directory.

    Args:
        argument: Argument to save
        arguments_dir: Base arguments directory

    Returns:
        Path to saved file
    """
    # Determine subdirectory based on source
    subdir = arguments_dir / argument.source
    ensure_dir(subdir)

    # Save to JSON file
    filepath = subdir / f"{argument.id}.json"
    with open(filepath, "w") as f:
        json.dump(argument.to_dict(), f, indent=2)

    return filepath


def load_argument(argument_id: str, arguments_dir: Path = ARGUMENTS_DIR) -> Argument | None:
    """
    Load an argument by ID.

    Args:
        argument_id: Argument ID to load
        arguments_dir: Base arguments directory

    Returns:
        Argument if found, None otherwise
    """
    # Search in all subdirectories
    for subdir in ["generated", "curated", "user"]:
        filepath = arguments_dir / subdir / f"{argument_id}.json"
        if filepath.exists():
            with open(filepath, "r") as f:
                return Argument.from_dict(json.load(f))

    return None


def load_arguments(
    arguments_dir: Path = ARGUMENTS_DIR,
    source: str | None = None,
    topic: str | None = None,
    stance: str | None = None,
    min_faults: int | None = None,
    max_faults: int | None = None,
) -> list[Argument]:
    """
    Load arguments with optional filtering.

    Args:
        arguments_dir: Base arguments directory
        source: Filter by source ("generated", "curated", "user")
        topic: Filter by topic ID
        stance: Filter by stance ("for", "against")
        min_faults: Minimum number of injected faults
        max_faults: Maximum number of injected faults

    Returns:
        List of matching Arguments
    """
    arguments = []

    # Determine which subdirectories to search
    subdirs = [source] if source else ["generated", "curated", "user"]

    for subdir_name in subdirs:
        subdir = arguments_dir / subdir_name
        if not subdir.exists():
            continue

        for filepath in subdir.glob("*.json"):
            with open(filepath, "r") as f:
                arg = Argument.from_dict(json.load(f))

            # Apply filters
            if topic and arg.topic != topic:
                continue
            if stance and arg.stance != stance:
                continue
            if min_faults is not None and len(arg.injected_faults) < min_faults:
                continue
            if max_faults is not None and len(arg.injected_faults) > max_faults:
                continue

            arguments.append(arg)

    return arguments


def list_argument_ids(arguments_dir: Path = ARGUMENTS_DIR) -> list[str]:
    """List all argument IDs."""
    ids = []
    for subdir_name in ["generated", "curated", "user"]:
        subdir = arguments_dir / subdir_name
        if subdir.exists():
            for filepath in subdir.glob("*.json"):
                ids.append(filepath.stem)
    return ids


def get_argument_index(arguments_dir: Path = ARGUMENTS_DIR) -> dict:
    """
    Build an index of all arguments with metadata.

    Returns:
        Dict mapping argument IDs to summary info
    """
    index = {}
    for arg in load_arguments(arguments_dir):
        index[arg.id] = {
            "topic": arg.topic,
            "stance": arg.stance,
            "source": arg.source,
            "fault_count": len(arg.injected_faults),
            "expected_score": arg.expected_score,
        }
    return index


# --- Experiment Results Storage ---

def save_results(results: ExperimentResults, results_dir: Path = RESULTS_DIR) -> Path:
    """
    Save experiment results.

    Args:
        results: ExperimentResults to save
        results_dir: Base results directory

    Returns:
        Path to experiment directory
    """
    experiment_dir = results_dir / results.experiment_id
    ensure_dir(experiment_dir)

    # Save config
    config_path = experiment_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(results.config.to_dict(), f, indent=2)

    # Save full results
    results_path = experiment_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)

    return experiment_dir


def load_results(experiment_id: str, results_dir: Path = RESULTS_DIR) -> ExperimentResults | None:
    """
    Load experiment results by ID.

    Args:
        experiment_id: Experiment ID to load
        results_dir: Base results directory

    Returns:
        ExperimentResults if found, None otherwise
    """
    results_path = results_dir / experiment_id / "results.json"
    if not results_path.exists():
        return None

    with open(results_path, "r") as f:
        return ExperimentResults.from_dict(json.load(f))


def list_experiments(results_dir: Path = RESULTS_DIR) -> list[str]:
    """List all experiment IDs."""
    if not results_dir.exists():
        return []

    return [
        d.name for d in results_dir.iterdir()
        if d.is_dir() and (d / "results.json").exists()
    ]


def append_evaluation(
    results: ExperimentResults,
    evaluation,
    results_dir: Path = RESULTS_DIR,
) -> None:
    """
    Append a single evaluation to results and save incrementally.
    Useful for crash recovery during long experiments.
    """
    results.add_evaluation(evaluation)
    save_results(results, results_dir)

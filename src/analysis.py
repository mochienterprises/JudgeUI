"""Analysis and reporting for experiment results."""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import statistics

from .config import load_faults, CONFIG_DIR
from .models import ExperimentResults, EvaluationResult
from .storage import load_argument, load_results, ARGUMENTS_DIR, RESULTS_DIR


@dataclass
class FaultDetectionStats:
    """Statistics for detecting a specific fault type."""
    fault_id: str
    total_injected: int = 0
    total_detected: int = 0
    false_positives: int = 0

    @property
    def detection_rate(self) -> float:
        if self.total_injected == 0:
            return 0.0
        return self.total_detected / self.total_injected

    @property
    def false_positive_rate(self) -> float:
        # FP rate relative to times it wasn't injected
        # This is a simplification - would need total non-injected count for true rate
        return self.false_positives


@dataclass
class ModelStats:
    """Aggregated statistics for a model configuration."""
    model: str
    temperature: float
    prompt: str

    scores: list[int]
    score_deltas: list[int]
    fault_stats: dict[str, FaultDetectionStats]

    @property
    def mean_score(self) -> float:
        return statistics.mean(self.scores) if self.scores else 0.0

    @property
    def std_score(self) -> float:
        return statistics.stdev(self.scores) if len(self.scores) > 1 else 0.0

    @property
    def mean_delta(self) -> float:
        return statistics.mean(self.score_deltas) if self.score_deltas else 0.0

    @property
    def std_delta(self) -> float:
        return statistics.stdev(self.score_deltas) if len(self.score_deltas) > 1 else 0.0


def analyze_experiment(
    results: ExperimentResults,
    arguments_dir: Path = ARGUMENTS_DIR,
    config_dir: Path = CONFIG_DIR,
) -> dict:
    """
    Analyze experiment results.

    Returns:
        Dict with analysis data including:
        - model_stats: Per-model/temp/prompt statistics
        - fault_detection: Per-fault detection rates by model
        - overall: Aggregate statistics
    """
    fault_configs = load_faults(config_dir)
    all_fault_ids = list(fault_configs.keys())

    # Group evaluations by model/temp/prompt combination
    grouped = defaultdict(list)
    for eval in results.evaluations:
        key = (eval.model, eval.temperature, eval.system_prompt)
        grouped[key].append(eval)

    # Calculate per-group statistics
    model_stats = {}
    for (model, temp, prompt), evals in grouped.items():
        key = f"{model}_{temp}_{prompt}"

        # Initialize fault stats
        fault_stats = {fid: FaultDetectionStats(fault_id=fid) for fid in all_fault_ids}

        scores = []
        deltas = []

        for eval in evals:
            scores.append(eval.score)
            deltas.append(eval.score_delta)

            # Load the argument to get injected faults
            arg = load_argument(eval.argument_id, arguments_dir)
            if arg:
                injected = set(arg.injected_faults)
                detected = set(eval.detected_faults)

                for fault_id in all_fault_ids:
                    if fault_id in injected:
                        fault_stats[fault_id].total_injected += 1
                        if fault_id in detected:
                            fault_stats[fault_id].total_detected += 1
                    elif fault_id in detected:
                        fault_stats[fault_id].false_positives += 1

        model_stats[key] = ModelStats(
            model=model,
            temperature=temp,
            prompt=prompt,
            scores=scores,
            score_deltas=deltas,
            fault_stats=fault_stats,
        )

    # Calculate overall fault detection rates
    overall_fault_stats = {fid: FaultDetectionStats(fault_id=fid) for fid in all_fault_ids}
    for stats in model_stats.values():
        for fid, fstats in stats.fault_stats.items():
            overall_fault_stats[fid].total_injected += fstats.total_injected
            overall_fault_stats[fid].total_detected += fstats.total_detected
            overall_fault_stats[fid].false_positives += fstats.false_positives

    return {
        "experiment_id": results.experiment_id,
        "config": results.config.to_dict(),
        "model_stats": model_stats,
        "overall_fault_stats": overall_fault_stats,
        "total_evaluations": len(results.evaluations),
    }


def generate_markdown_report(
    results: ExperimentResults,
    arguments_dir: Path = ARGUMENTS_DIR,
    config_dir: Path = CONFIG_DIR,
) -> str:
    """Generate a markdown analysis report."""
    analysis = analyze_experiment(results, arguments_dir, config_dir)

    lines = [
        f"# Experiment Analysis: {results.experiment_id}",
        "",
        f"**Started:** {results.started_at}",
        f"**Completed:** {results.completed_at or 'In progress'}",
        f"**Total Evaluations:** {analysis['total_evaluations']}",
        "",
        "## Configuration",
        "",
        f"- **Models:** {', '.join(results.config.models)}",
        f"- **Temperatures:** {', '.join(map(str, results.config.temperatures))}",
        f"- **Evaluator Prompts:** {', '.join(results.config.evaluator_prompts)}",
        f"- **Runs per Combination:** {results.config.runs_per_combination}",
        "",
    ]

    # Score calibration by model
    lines.extend([
        "## Score Calibration by Model",
        "",
        "| Model | Temp | Prompt | Mean Score | Std Dev | Mean Delta | Delta Std |",
        "|-------|------|--------|------------|---------|------------|-----------|",
    ])

    for key, stats in sorted(analysis["model_stats"].items()):
        lines.append(
            f"| {stats.model} | {stats.temperature} | {stats.prompt} | "
            f"{stats.mean_score:.1f} | {stats.std_score:.1f} | "
            f"{stats.mean_delta:+.1f} | {stats.std_delta:.1f} |"
        )

    lines.append("")

    # Fault detection rates
    lines.extend([
        "## Overall Fault Detection Rates",
        "",
        "| Fault | Injected | Detected | Detection Rate | False Positives |",
        "|-------|----------|----------|----------------|-----------------|",
    ])

    for fault_id, stats in sorted(
        analysis["overall_fault_stats"].items(),
        key=lambda x: x[1].detection_rate,
        reverse=True,
    ):
        if stats.total_injected > 0 or stats.false_positives > 0:
            lines.append(
                f"| {fault_id} | {stats.total_injected} | {stats.total_detected} | "
                f"{stats.detection_rate:.1%} | {stats.false_positives} |"
            )

    lines.append("")

    # Fault detection by model
    lines.extend([
        "## Fault Detection by Model",
        "",
    ])

    # Get models with data
    model_keys = sorted(analysis["model_stats"].keys())

    if model_keys:
        # Header
        header = "| Fault |"
        separator = "|-------|"
        for key in model_keys:
            header += f" {key} |"
            separator += "--------|"
        lines.append(header)
        lines.append(separator)

        # Rows for each fault
        fault_ids = sorted(set(
            fid for stats in analysis["model_stats"].values()
            for fid in stats.fault_stats.keys()
            if stats.fault_stats[fid].total_injected > 0
        ))

        for fault_id in fault_ids:
            row = f"| {fault_id} |"
            for key in model_keys:
                stats = analysis["model_stats"][key].fault_stats.get(fault_id)
                if stats and stats.total_injected > 0:
                    row += f" {stats.detection_rate:.0%} |"
                else:
                    row += " - |"
            lines.append(row)

    lines.append("")

    # Consistency analysis (std dev by temperature)
    lines.extend([
        "## Consistency Analysis (Score Std Dev by Temperature)",
        "",
    ])

    temp_consistency = defaultdict(list)
    for stats in analysis["model_stats"].values():
        temp_consistency[stats.temperature].append(stats.std_score)

    lines.append("| Temperature | Avg Std Dev | Min | Max |")
    lines.append("|-------------|-------------|-----|-----|")

    for temp in sorted(temp_consistency.keys()):
        stds = temp_consistency[temp]
        lines.append(
            f"| {temp} | {statistics.mean(stds):.1f} | "
            f"{min(stds):.1f} | {max(stds):.1f} |"
        )

    lines.append("")
    lines.append("---")
    lines.append(f"*Generated at {datetime.now().isoformat()}*")

    return "\n".join(lines)


def save_analysis_report(
    results: ExperimentResults,
    results_dir: Path = RESULTS_DIR,
    arguments_dir: Path = ARGUMENTS_DIR,
    config_dir: Path = CONFIG_DIR,
) -> Path:
    """Generate and save analysis report to experiment directory."""
    report = generate_markdown_report(results, arguments_dir, config_dir)

    report_path = results_dir / results.experiment_id / "analysis.md"
    with open(report_path, "w") as f:
        f.write(report)

    return report_path


def compare_experiments(
    experiment_ids: list[str],
    results_dir: Path = RESULTS_DIR,
) -> str:
    """Generate a comparison report across multiple experiments."""
    lines = [
        "# Experiment Comparison",
        "",
        f"**Experiments:** {', '.join(experiment_ids)}",
        f"**Generated:** {datetime.now().isoformat()}",
        "",
    ]

    # Load all results
    all_results = []
    for exp_id in experiment_ids:
        results = load_results(exp_id, results_dir)
        if results:
            all_results.append(results)
        else:
            lines.append(f"Warning: Could not load experiment {exp_id}")

    if not all_results:
        return "\n".join(lines + ["No experiments found."])

    # Summary table
    lines.extend([
        "## Summary",
        "",
        "| Experiment | Models | Evaluations | Date |",
        "|------------|--------|-------------|------|",
    ])

    for results in all_results:
        lines.append(
            f"| {results.experiment_id} | {len(results.config.models)} | "
            f"{len(results.evaluations)} | {results.started_at[:10]} |"
        )

    lines.append("")

    return "\n".join(lines)

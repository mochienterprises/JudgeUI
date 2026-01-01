#!/usr/bin/env python3
"""CLI for analyzing experiment results."""

import argparse
import sys
from pathlib import Path

from src.analysis import (
    analyze_experiment,
    generate_markdown_report,
    save_analysis_report,
    compare_experiments,
)
from src.storage import load_results, list_experiments, RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Analyze experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a specific experiment
  python analyze.py my-experiment_20250101_120000

  # List all experiments
  python analyze.py --list

  # Compare multiple experiments
  python analyze.py --compare exp1 exp2 exp3

  # Print analysis to stdout instead of saving
  python analyze.py my-experiment --stdout
        """
    )

    parser.add_argument("experiment_id", nargs="?", help="Experiment ID to analyze")
    parser.add_argument("--list", action="store_true", help="List all experiments")
    parser.add_argument("--compare", nargs="+", metavar="EXP_ID", help="Compare multiple experiments")
    parser.add_argument("--stdout", action="store_true", help="Print report to stdout instead of saving")
    parser.add_argument("--json", action="store_true", help="Output raw analysis data as JSON")

    args = parser.parse_args()

    if args.list:
        print("\nExperiments:")
        print("-" * 60)
        exp_ids = list_experiments()
        if exp_ids:
            for exp_id in sorted(exp_ids):
                results = load_results(exp_id)
                if results:
                    status = "complete" if results.completed_at else "in progress"
                    print(f"  {exp_id}")
                    print(f"    Evaluations: {len(results.evaluations)}, Status: {status}")
        else:
            print("  No experiments found.")
        return

    if args.compare:
        report = compare_experiments(args.compare)
        print(report)
        return

    if not args.experiment_id:
        parser.print_help()
        print("\nError: Provide an experiment_id or use --list")
        sys.exit(1)

    # Load experiment
    results = load_results(args.experiment_id)
    if not results:
        print(f"Error: Experiment not found: {args.experiment_id}")
        print("\nAvailable experiments:")
        for exp_id in list_experiments():
            print(f"  {exp_id}")
        sys.exit(1)

    if args.json:
        import json
        analysis = analyze_experiment(results)
        # Convert non-serializable objects
        output = {
            "experiment_id": analysis["experiment_id"],
            "config": analysis["config"],
            "total_evaluations": analysis["total_evaluations"],
            "model_stats": {
                key: {
                    "model": stats.model,
                    "temperature": stats.temperature,
                    "prompt": stats.prompt,
                    "mean_score": stats.mean_score,
                    "std_score": stats.std_score,
                    "mean_delta": stats.mean_delta,
                    "std_delta": stats.std_delta,
                }
                for key, stats in analysis["model_stats"].items()
            },
            "fault_detection": {
                fid: {
                    "total_injected": stats.total_injected,
                    "total_detected": stats.total_detected,
                    "detection_rate": stats.detection_rate,
                    "false_positives": stats.false_positives,
                }
                for fid, stats in analysis["overall_fault_stats"].items()
                if stats.total_injected > 0 or stats.false_positives > 0
            },
        }
        print(json.dumps(output, indent=2))
        return

    # Generate report
    report = generate_markdown_report(results)

    if args.stdout:
        print(report)
    else:
        report_path = save_analysis_report(results)
        print(f"Analysis report saved to: {report_path}")
        print("\nSummary:")
        print(f"  Experiment: {results.experiment_id}")
        print(f"  Total evaluations: {len(results.evaluations)}")
        print(f"  Status: {'Complete' if results.completed_at else 'In progress'}")


if __name__ == "__main__":
    main()

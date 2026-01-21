#!/usr/bin/env python3
"""CLI for running experiments."""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

from src.experiment import run_experiment_from_file, run_experiment
from src.models import ExperimentConfig
from src.storage import load_arguments, list_argument_ids, list_experiments


def list_available():
    """Print available arguments and experiments."""
    print("\nAvailable Arguments:")
    print("-" * 40)
    arg_ids = list_argument_ids()
    if arg_ids:
        for arg_id in sorted(arg_ids):
            print(f"  {arg_id}")
    else:
        print("  No arguments found. Use generate_arguments.py to create some.")

    print("\nExisting Experiments:")
    print("-" * 40)
    exp_ids = list_experiments()
    if exp_ids:
        for exp_id in sorted(exp_ids):
            print(f"  {exp_id}")
    else:
        print("  No experiments found.")


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run experiment from YAML file
  python run_experiment.py experiments/example.yaml

  # Quick test: evaluate all arguments with one model
  python run_experiment.py --quick --model claude-sonnet-4-5

  # Custom experiment from command line
  python run_experiment.py --name "temp-test" \\
      --models claude-sonnet-4-5,gpt-4o \\
      --temperatures 0.0,0.5,1.0 \\
      --prompts default,strict

  # List available arguments and experiments
  python run_experiment.py --list
        """
    )

    parser.add_argument("experiment_file", nargs="?", help="Path to experiment YAML file")

    parser.add_argument("--list", action="store_true", help="List available arguments and experiments")

    # Quick test options
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: evaluate all arguments with specified model")
    parser.add_argument("--model", help="Model for quick test")

    # Custom experiment options
    parser.add_argument("--name", help="Experiment name (for custom experiments)")
    parser.add_argument("--models", help="Comma-separated model IDs")
    parser.add_argument("--temperatures", help="Comma-separated temperatures")
    parser.add_argument("--prompts", help="Comma-separated evaluator prompt names")
    parser.add_argument("--argument-ids", help="Comma-separated argument IDs (default: all)")
    parser.add_argument("--runs", type=int, default=1, help="Runs per combination (default: 1)")
    parser.add_argument("--parallel", "-p", action="store_true",
                        help="Run evaluations in parallel across models (faster)")

    args = parser.parse_args()

    if args.list:
        list_available()
        return

    # Determine experiment mode
    if args.experiment_file:
        # Run from YAML file
        experiment_path = Path(args.experiment_file)
        if not experiment_path.exists():
            print(f"Error: Experiment file not found: {experiment_path}")
            sys.exit(1)

        print(f"Running experiment from: {experiment_path}")
        results = run_experiment_from_file(experiment_path, parallel=args.parallel)

    elif args.quick:
        # Quick test mode
        if not args.model:
            parser.error("--model is required for quick test")

        arg_ids = list_argument_ids()
        if not arg_ids:
            print("Error: No arguments found. Use generate_arguments.py first.")
            sys.exit(1)

        config = ExperimentConfig(
            name=f"quick-{args.model}",
            description="Quick test run",
            models=[args.model],
            temperatures=[1.0],
            evaluator_prompts=["default"],
            argument_ids=arg_ids,
            runs_per_combination=1,
        )

        print(f"Quick test with {args.model} on {len(arg_ids)} arguments")
        results = run_experiment(config, parallel=args.parallel)

    elif args.name:
        # Custom experiment from CLI args
        if not args.models:
            parser.error("--models is required for custom experiments")

        arg_ids = args.argument_ids.split(",") if args.argument_ids else list_argument_ids()
        if not arg_ids:
            print("Error: No arguments found. Use generate_arguments.py first.")
            sys.exit(1)

        config = ExperimentConfig(
            name=args.name,
            description="Custom experiment",
            models=args.models.split(","),
            temperatures=[float(t) for t in args.temperatures.split(",")] if args.temperatures else [1.0],
            evaluator_prompts=args.prompts.split(",") if args.prompts else ["default"],
            argument_ids=arg_ids,
            runs_per_combination=args.runs,
        )

        results = run_experiment(config, parallel=args.parallel)

    else:
        parser.print_help()
        print("\nError: Provide an experiment file, use --quick, or specify --name for custom experiment")
        sys.exit(1)

    # Generate analysis report
    from src.analysis import save_analysis_report
    report_path = save_analysis_report(results)
    print(f"Analysis report saved to: {report_path}")


if __name__ == "__main__":
    main()

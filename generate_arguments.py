#!/usr/bin/env python3
"""CLI for generating arguments with fault injection."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

from src.config import load_faults, load_topics, CONFIG_DIR
from src.generator import generate_argument, generate_argument_set
from src.providers import AnthropicProvider, OpenAIProvider
from src.storage import save_argument, load_arguments, get_argument_index


def list_faults():
    """Print available fault types."""
    faults = load_faults()
    print("\nAvailable Faults:")
    print("-" * 60)

    current_category = None
    for fault_id, fault in sorted(faults.items(), key=lambda x: (x[1].category, x[0])):
        if fault.category != current_category:
            current_category = fault.category
            print(f"\n  {current_category.upper()}")

        print(f"    {fault_id}: {fault.description} (severity: {fault.severity})")


def list_topics():
    """Print available topics."""
    topics = load_topics()
    print("\nAvailable Topics:")
    print("-" * 60)

    current_category = None
    for topic_id, topic in sorted(topics.items(), key=lambda x: (x[1].category, x[0])):
        if topic.category != current_category:
            current_category = topic.category
            print(f"\n  {current_category.upper()}")

        print(f"    {topic_id}: {topic.title}")


def list_arguments():
    """Print existing arguments."""
    index = get_argument_index()
    print(f"\nExisting Arguments: {len(index)}")
    print("-" * 80)
    print(f"{'ID':<10} {'Topic':<25} {'Stance':<10} {'Faults':<8} {'Expected':<10} {'Source'}")
    print("-" * 80)

    for arg_id, info in sorted(index.items()):
        print(
            f"{arg_id:<10} {info['topic']:<25} {info['stance']:<10} "
            f"{info['fault_count']:<8} {info['expected_score']:<10} {info['source']}"
        )


def get_provider(provider_name: str):
    """Get provider instance by name."""
    if provider_name == "anthropic":
        return AnthropicProvider()
    elif provider_name == "openai":
        return OpenAIProvider()
    else:
        raise ValueError(f"Unknown provider: {provider_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate debate arguments with optional fault injection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a clean argument
  python generate_arguments.py --topic gun-control --stance for

  # Generate with specific faults
  python generate_arguments.py --topic gun-control --stance for --faults ad_hominem,no_evidence

  # Generate a full set (clean + light + heavy faults for both stances)
  python generate_arguments.py --topic gun-control --full-set

  # List available faults
  python generate_arguments.py --list-faults

  # List available topics
  python generate_arguments.py --list-topics
        """
    )

    parser.add_argument("--topic", "-t", help="Topic ID from topics.yaml or custom topic text")
    parser.add_argument("--stance", "-s", choices=["for", "against"], help="Argument stance")
    parser.add_argument("--faults", "-f", help="Comma-separated list of fault IDs to inject")
    parser.add_argument("--provider", "-p", default="anthropic", choices=["anthropic", "openai"],
                        help="LLM provider (default: anthropic)")
    parser.add_argument("--model", "-m", help="Model ID to use (provider default if not specified)")
    parser.add_argument("--full-set", action="store_true",
                        help="Generate full set: clean, light, heavy faults for both stances")

    parser.add_argument("--list-faults", action="store_true", help="List available fault types")
    parser.add_argument("--list-topics", action="store_true", help="List available topics")
    parser.add_argument("--list-arguments", action="store_true", help="List existing arguments")

    args = parser.parse_args()

    # Handle list commands
    if args.list_faults:
        list_faults()
        return

    if args.list_topics:
        list_topics()
        return

    if args.list_arguments:
        list_arguments()
        return

    # Validate arguments for generation
    if not args.topic:
        parser.error("--topic is required for generation")

    if not args.full_set and not args.stance:
        parser.error("--stance is required (or use --full-set)")

    # Resolve topic
    topics = load_topics()
    if args.topic in topics:
        topic_title = topics[args.topic].title
    else:
        topic_title = args.topic  # Use as-is if not in library

    print(f"\nTopic: {topic_title}")
    print(f"Provider: {args.provider}")

    try:
        provider = get_provider(args.provider)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.full_set:
        # Generate full set
        print("\nGenerating full argument set...")
        arguments = generate_argument_set(
            provider=provider,
            topic=topic_title,
            model=args.model,
        )

        print(f"\nGenerated {len(arguments)} arguments:")
        for arg in arguments:
            path = save_argument(arg)
            print(f"  [{arg.id}] {arg.stance.upper()} - {len(arg.injected_faults)} faults - saved to {path}")

    else:
        # Generate single argument
        faults = args.faults.split(",") if args.faults else []

        print(f"Stance: {args.stance}")
        print(f"Faults: {faults if faults else 'None (clean argument)'}")
        print("\nGenerating argument...")

        argument = generate_argument(
            provider=provider,
            topic=topic_title,
            stance=args.stance,
            faults=faults,
            model=args.model,
        )

        path = save_argument(argument)

        print(f"\n{'=' * 60}")
        print(f"ID: {argument.id}")
        print(f"Expected Score: {argument.expected_score}")
        print(f"Saved to: {path}")
        print(f"{'=' * 60}")
        print(f"\n{argument.text}")


if __name__ == "__main__":
    main()

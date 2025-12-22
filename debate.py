import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize clients
anthropic_client = None
openai_client = None

if os.getenv("ANTHROPIC_API_KEY"):
    anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

if os.getenv("OPENAI_API_KEY"):
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Constants
MAX_ARGUMENT_LENGTH = 5000
ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"
OPENAI_MODEL = "gpt-4o"

# Global provider selection
selected_provider = None


def call_llm(prompt, max_tokens=2000):
    """Unified function to call either Anthropic or OpenAI based on selected provider."""

    if selected_provider == "anthropic":
        if not anthropic_client:
            raise ValueError("Anthropic API key not found in .env file")

        response = anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()

    elif selected_provider == "openai":
        if not openai_client:
            raise ValueError("OpenAI API key not found in .env file")

        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()

    else:
        raise ValueError("No provider selected")


def generate_argument(topic, stance, quality):
    """Generate a single argument for the topic with specified quality level."""

    quality_instructions = {
        "low": """Create a LOW quality argument with:
- Multiple logical fallacies (ad hominem, strawman, false dichotomy, etc.)
- Heavy emotional appeals instead of logic
- Anecdotal evidence presented as fact
- Strawman the opponent's position
- Poor reasoning and weak connections between points
- Maximum length: 5000 characters""",

        "medium": """Create a MEDIUM quality argument with:
- Some valid logical points but weak or limited evidence
- Minor gaps in reasoning or logical leaps
- A few hasty generalizations
- Some appeal to authority without strong backing
- Decent structure but not thoroughly developed
- Maximum length: 5000 characters""",

        "high": """Create a HIGH quality argument with:
- Strong logical reasoning and well-structured points
- Cites credible evidence and research
- Steelmans the opponent's position before addressing it
- Acknowledges nuance and complexity
- Anticipates and addresses counterarguments
- Avoids logical fallacies
- Maximum length: 5000 characters"""
    }

    stance_text = "FOR" if stance == "for" else "AGAINST"

    prompt = f"""You are generating an argument {stance_text} the following topic:
"{topic}"

{quality_instructions[quality]}

Write only the argument itself, no preamble or explanation. The argument should be persuasive and match the quality level specified."""

    argument = call_llm(prompt, max_tokens=2000)

    # Enforce character limit
    if len(argument) > MAX_ARGUMENT_LENGTH:
        argument = argument[:MAX_ARGUMENT_LENGTH]

    return argument


def evaluate_argument_blind(argument):
    """Evaluate an argument without knowing the topic or stance."""

    prompt = f"""Evaluate the following argument without knowing what topic it's about.

Provide your evaluation in JSON format with:
- score: integer from 0-100
- reasoning: brief explanation of the score
- fallacies: list of fallacies detected (empty list if none)

Argument to evaluate:
{argument}

Respond with ONLY valid JSON, no other text."""

    result_text = call_llm(prompt, max_tokens=1000)

    # Clean up JSON if wrapped in markdown code blocks
    if result_text.startswith("```"):
        lines = result_text.split("\n")
        result_text = "\n".join(lines[1:-1]) if len(lines) > 2 else result_text
        if result_text.startswith("json"):
            result_text = result_text[4:].strip()

    try:
        evaluation = json.loads(result_text)
        return evaluation
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        return {
            "score": 50,
            "reasoning": "Error parsing evaluation",
            "fallacies": []
        }


def display_results(results):
    """Display results in a formatted table."""

    print("\n" + "="*100)
    print("EVALUATION RESULTS")
    print("="*100)
    print(f"{'Stance':<10} {'Quality':<10} {'Score':<10} {'Fallacies':<20} {'Reasoning'}")
    print("-"*100)

    for r in results:
        fallacies_str = ", ".join(r['evaluation']['fallacies'][:3]) if r['evaluation']['fallacies'] else "None"
        if len(fallacies_str) > 18:
            fallacies_str = fallacies_str[:15] + "..."

        reasoning = r['evaluation']['reasoning']
        if len(reasoning) > 50:
            reasoning = reasoning[:47] + "..."

        print(f"{r['stance']:<10} {r['quality']:<10} {r['evaluation']['score']:<10} {fallacies_str:<20} {reasoning}")

    print("="*100)


def check_ranking(results):
    """Check if quality rankings are correct within each stance."""

    print("\n" + "="*80)
    print("RANKING ANALYSIS")
    print("="*80)

    for_results = [r for r in results if r['stance'] == 'FOR']
    against_results = [r for r in results if r['stance'] == 'AGAINST']

    def analyze_stance_ranking(stance_results, stance_name):
        low = next(r for r in stance_results if r['quality'] == 'low')
        medium = next(r for r in stance_results if r['quality'] == 'medium')
        high = next(r for r in stance_results if r['quality'] == 'high')

        low_score = low['evaluation']['score']
        med_score = medium['evaluation']['score']
        high_score = high['evaluation']['score']

        print(f"\n{stance_name}:")
        print(f"  Low quality:    {low_score}")
        print(f"  Medium quality: {med_score}")
        print(f"  High quality:   {high_score}")

        correct = low_score < med_score < high_score

        if correct:
            print(f"  ✓ Ranking is CORRECT (low < medium < high)")
        else:
            print(f"  ✗ Ranking is INCORRECT")
            if low_score >= med_score:
                print(f"    - Low ({low_score}) should be < Medium ({med_score})")
            if med_score >= high_score:
                print(f"    - Medium ({med_score}) should be < High ({high_score})")

        return correct

    for_correct = analyze_stance_ranking(for_results, "FOR arguments")
    against_correct = analyze_stance_ranking(against_results, "AGAINST arguments")

    print(f"\nOverall ranking: {'✓ CORRECT' if for_correct and against_correct else '✗ INCORRECT'}")
    print("="*80)

    return for_correct and against_correct


def check_bias(results):
    """Check for bias between FOR and AGAINST arguments at same quality levels."""

    print("\n" + "="*80)
    print("BIAS ANALYSIS")
    print("="*80)

    for_results = {r['quality']: r['evaluation']['score'] for r in results if r['stance'] == 'FOR'}
    against_results = {r['quality']: r['evaluation']['score'] for r in results if r['stance'] == 'AGAINST'}

    print(f"\n{'Quality':<15} {'FOR Score':<15} {'AGAINST Score':<15} {'Difference':<15} {'Status'}")
    print("-"*80)

    total_diff = 0

    for quality in ['low', 'medium', 'high']:
        for_score = for_results[quality]
        against_score = against_results[quality]
        diff = abs(for_score - against_score)
        total_diff += diff

        status = "Balanced" if diff <= 10 else "Potential bias"

        print(f"{quality.capitalize():<15} {for_score:<15} {against_score:<15} {diff:<15} {status}")

    avg_diff = total_diff / 3

    print("-"*80)
    print(f"Average difference: {avg_diff:.1f}")

    if avg_diff <= 10:
        print("✓ No significant bias detected (avg diff <= 10 points)")
    else:
        print("✗ Potential bias detected (avg diff > 10 points)")

    print("="*80)

    return avg_diff


def view_arguments(results):
    """Display all generated arguments."""

    print("\n" + "="*100)
    print("GENERATED ARGUMENTS")
    print("="*100)

    for r in results:
        print(f"\n[{r['stance']} - {r['quality'].upper()} quality]")
        print("-"*100)
        print(r['argument'])
        print("-"*100)


def save_to_markdown(topic, results, ranking_correct, avg_bias, provider):
    """Save results to a markdown file."""

    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = results_dir / f"debate_{timestamp}.md"

    # Build markdown content
    md_content = f"""# Debate Evaluation Results

**Topic:** {topic}
**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Provider:** {provider.upper()}
**Model:** {ANTHROPIC_MODEL if provider == 'anthropic' else OPENAI_MODEL}

## Summary

- **Ranking Correct:** {'Yes' if ranking_correct else 'No'}
- **Average Bias:** {avg_bias:.1f} points

## Detailed Results

| Stance | Quality | Score | Fallacies | Reasoning |
|--------|---------|-------|-----------|-----------|
"""

    for r in results:
        fallacies = ", ".join(r['evaluation']['fallacies']) if r['evaluation']['fallacies'] else "None"
        md_content += f"| {r['stance']} | {r['quality']} | {r['evaluation']['score']} | {fallacies} | {r['evaluation']['reasoning']} |\n"

    md_content += "\n## Generated Arguments\n\n"

    for r in results:
        md_content += f"### {r['stance']} - {r['quality'].upper()} Quality\n\n"
        md_content += f"{r['argument']}\n\n"
        md_content += "---\n\n"

    # Write to file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"\n✓ Results saved to: {filename}")


def select_provider():
    """Ask user to select which AI provider to use."""
    global selected_provider

    available_providers = []

    if anthropic_client:
        available_providers.append("anthropic")
    if openai_client:
        available_providers.append("openai")

    if not available_providers:
        print("Error: No API keys found in .env file")
        print("Please add either ANTHROPIC_API_KEY or OPENAI_API_KEY to your .env file")
        return False

    if len(available_providers) == 1:
        selected_provider = available_providers[0]
        print(f"\nUsing {selected_provider.upper()} (only provider with API key configured)")
        return True

    print("\nAvailable AI providers:")
    print("1. Anthropic (Claude Sonnet 4.5)")
    print("2. OpenAI (GPT-4o)")

    while True:
        choice = input("\nSelect provider (1 or 2): ").strip()

        if choice == "1":
            selected_provider = "anthropic"
            break
        elif choice == "2":
            selected_provider = "openai"
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

    print(f"Using {selected_provider.upper()}")
    return True


def main():
    print("="*80)
    print("DEBATE EVALUATION TOOL")
    print("="*80)

    # Select AI provider
    if not select_provider():
        return

    # Get topic from user
    topic = input("\nEnter debate topic: ").strip()

    if not topic:
        print("Error: Topic cannot be empty")
        return

    print(f"\nTopic: {topic}")
    print("\nGenerating arguments...")

    # Generate all 6 arguments
    results = []

    for stance in ['for', 'against']:
        for quality in ['low', 'medium', 'high']:
            stance_label = stance.upper()
            print(f"  Generating {stance_label} - {quality} quality...")

            argument = generate_argument(topic, stance, quality)

            results.append({
                'stance': stance_label,
                'quality': quality,
                'argument': argument,
                'evaluation': None
            })

    print("\nEvaluating arguments (blind)...")

    # Evaluate each argument
    for i, r in enumerate(results):
        print(f"  Evaluating argument {i+1}/6...")
        r['evaluation'] = evaluate_argument_blind(r['argument'])

    # Display results
    display_results(results)

    # Check ranking
    ranking_correct = check_ranking(results)

    # Check bias
    avg_bias = check_bias(results)

    # Ask if user wants to view arguments
    print("\n" + "="*80)
    view = input("\nView generated arguments? (y/n): ").strip().lower()
    if view == 'y':
        view_arguments(results)

    # Ask if user wants to save results
    save = input("\nSave results to markdown file? (y/n): ").strip().lower()
    if save == 'y':
        save_to_markdown(topic, results, ranking_correct, avg_bias, selected_provider)

    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == "__main__":
    main()

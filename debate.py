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
AUTO_WIN_THRESHOLD = 5  # Score difference threshold for automatic wins

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


def determine_winner(results, topic):
    """
    Determine the winner between best FOR and AGAINST arguments.

    Args:
        results: List of result dictionaries with stance, quality, argument, evaluation
        topic: The debate topic string

    Returns:
        Dictionary with winner information
    """
    # Extract best arguments from each stance
    for_args = [r for r in results if r['stance'] == 'FOR']
    against_args = [r for r in results if r['stance'] == 'AGAINST']

    best_for = max(for_args, key=lambda x: x['evaluation']['score'])
    best_against = max(against_args, key=lambda x: x['evaluation']['score'])

    for_score = best_for['evaluation']['score']
    against_score = best_against['evaluation']['score']
    score_diff = abs(for_score - against_score)

    # Determine winner
    if score_diff > AUTO_WIN_THRESHOLD:
        # Automatic win - score difference is decisive
        winner = 'FOR' if for_score > against_score else 'AGAINST'
        reasoning = f"Clear victory based on evaluation scores. The {winner} argument scored {max(for_score, against_score)} points compared to {min(for_score, against_score)} points for the opposing side, a difference of {score_diff} points."

        return {
            'method': 'automatic',
            'winner': winner,
            'for_score': for_score,
            'against_score': against_score,
            'score_difference': score_diff,
            'reasoning': reasoning,
            'for_argument': best_for['argument'],
            'against_argument': best_against['argument'],
            'for_quality': best_for['quality'],
            'against_quality': best_against['quality']
        }
    else:
        # Scores within threshold - call LLM for head-to-head comparison
        print(f"  Scores within {AUTO_WIN_THRESHOLD} points ({for_score} vs {against_score})")
        print("  Conducting head-to-head evaluation...")

        llm_result = evaluate_winner(
            topic,
            best_for['argument'],
            for_score,
            best_against['argument'],
            against_score
        )

        # Merge LLM result with base information
        return {
            'method': 'head-to-head',
            'for_score': for_score,
            'against_score': against_score,
            'score_difference': score_diff,
            'for_argument': best_for['argument'],
            'against_argument': best_against['argument'],
            'for_quality': best_for['quality'],
            'against_quality': best_against['quality'],
            **llm_result  # Adds winner, margin, reasoning, for_strengths, against_strengths, deciding_factor
        }


def evaluate_winner(topic, for_arg, for_score, against_arg, against_score):
    """
    Perform head-to-head LLM evaluation when scores are within threshold.

    Args:
        topic: Debate topic string
        for_arg: Best FOR argument text
        for_score: Blind evaluation score for FOR argument
        against_arg: Best AGAINST argument text
        against_score: Blind evaluation score for AGAINST argument

    Returns:
        Dictionary with winner, margin, reasoning, strengths, and deciding_factor
    """
    prompt = f"""You are a debate judge. You must select a winner.

TOPIC: "{topic}"

ARGUMENT FOR:
{for_arg}
(Quality Score: {for_score})

ARGUMENT AGAINST:
{against_arg}
(Quality Score: {against_score})

Evaluate which argument better addresses the topic. Consider:
1. Which argument more directly answers the question posed by the topic?
2. Which would be more convincing to a neutral, intelligent audience?
3. Which makes the stronger overall case for their position?

You MUST choose a winner. No ties allowed.

Respond in JSON:
{{
  "winner": "FOR" or "AGAINST",
  "margin": "narrow" or "moderate" or "decisive",
  "reasoning": "<2-3 sentences explaining why this argument wins>",
  "for_strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "against_strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "deciding_factor": "<what tipped the scales>"
}}

Margin definitions:
- "narrow": Very close, could go either way
- "moderate": Clear advantage but opponent had merit
- "decisive": Overwhelming superiority

Respond with ONLY valid JSON, no other text."""

    result_text = call_llm(prompt, max_tokens=1500)

    # Clean up JSON if wrapped in markdown code blocks
    if result_text.startswith("```"):
        lines = result_text.split("\n")
        result_text = "\n".join(lines[1:-1]) if len(lines) > 2 else result_text
        if result_text.startswith("json"):
            result_text = result_text[4:].strip()

    try:
        evaluation = json.loads(result_text)

        # Validate required fields
        required_fields = ['winner', 'margin', 'reasoning', 'for_strengths', 'against_strengths', 'deciding_factor']
        if not all(field in evaluation for field in required_fields):
            raise ValueError("Missing required fields in JSON response")

        # Validate winner value
        if evaluation['winner'] not in ['FOR', 'AGAINST']:
            raise ValueError(f"Invalid winner value: {evaluation['winner']}")

        # Validate margin value
        if evaluation['margin'] not in ['narrow', 'moderate', 'decisive']:
            raise ValueError(f"Invalid margin value: {evaluation['margin']}")

        return evaluation

    except (json.JSONDecodeError, ValueError) as e:
        # Fallback if JSON parsing or validation fails
        print(f"  Warning: Error parsing winner evaluation ({e}), using fallback")

        # Determine winner based on scores as fallback
        winner = 'FOR' if for_score >= against_score else 'AGAINST'

        return {
            'winner': winner,
            'margin': 'narrow',
            'reasoning': f"Based on blind evaluation scores. FOR: {for_score}, AGAINST: {against_score}.",
            'for_strengths': ["Unable to determine (evaluation error)"],
            'against_strengths': ["Unable to determine (evaluation error)"],
            'deciding_factor': "Score-based fallback due to evaluation error"
        }


def display_winner(winner_result):
    """
    Display winner determination in terminal.

    Args:
        winner_result: Dictionary from determine_winner()
    """
    print("\n" + "="*80)
    print("WINNER DETERMINATION")
    print("="*80)

    print(f"\nMethod: {winner_result['method'].upper()}")
    print(f"\nBest FOR argument: {winner_result['for_quality']} quality (Score: {winner_result['for_score']})")
    print(f"Best AGAINST argument: {winner_result['against_quality']} quality (Score: {winner_result['against_score']})")
    print(f"Score difference: {winner_result['score_difference']} points")

    print(f"\n{'='*80}")
    print(f"WINNER: {winner_result['winner']}")

    if winner_result['method'] == 'head-to-head':
        print(f"Margin: {winner_result['margin'].upper()}")

    print(f"{'='*80}")

    print(f"\nReasoning:")
    print(f"  {winner_result['reasoning']}")

    if winner_result['method'] == 'head-to-head':
        print(f"\nFOR Argument Strengths:")
        for strength in winner_result['for_strengths']:
            print(f"  - {strength}")

        print(f"\nAGAINST Argument Strengths:")
        for strength in winner_result['against_strengths']:
            print(f"  - {strength}")

        print(f"\nDeciding Factor:")
        print(f"  {winner_result['deciding_factor']}")

    print("="*80)


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


def save_to_markdown(topic, results, ranking_correct, avg_bias, provider, winner_result):
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

    # Add Winner Determination section
    md_content += "## Winner Determination\n\n"

    md_content += f"**Method:** {winner_result['method'].capitalize()}\n\n"
    md_content += f"**Best FOR Argument:** {winner_result['for_quality']} quality (Score: {winner_result['for_score']})\n\n"
    md_content += f"**Best AGAINST Argument:** {winner_result['against_quality']} quality (Score: {winner_result['against_score']})\n\n"
    md_content += f"**Score Difference:** {winner_result['score_difference']} points\n\n"

    md_content += f"### Winner: {winner_result['winner']}\n\n"

    if winner_result['method'] == 'head-to-head':
        md_content += f"**Margin:** {winner_result['margin'].capitalize()}\n\n"

    md_content += f"**Reasoning:** {winner_result['reasoning']}\n\n"

    if winner_result['method'] == 'head-to-head':
        md_content += "#### FOR Argument Strengths\n\n"
        for strength in winner_result['for_strengths']:
            md_content += f"- {strength}\n"

        md_content += "\n#### AGAINST Argument Strengths\n\n"
        for strength in winner_result['against_strengths']:
            md_content += f"- {strength}\n"

        md_content += f"\n**Deciding Factor:** {winner_result['deciding_factor']}\n\n"

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

    # Determine winner
    print("\nDetermining winner...")
    winner_result = determine_winner(results, topic)

    # Display winner
    display_winner(winner_result)

    # Ask if user wants to view arguments
    print("\n" + "="*80)
    view = input("\nView generated arguments? (y/n): ").strip().lower()
    if view == 'y':
        view_arguments(results)

    # Ask if user wants to save results
    save = input("\nSave results to markdown file? (y/n): ").strip().lower()
    if save == 'y':
        save_to_markdown(topic, results, ranking_correct, avg_bias, selected_provider, winner_result)

    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == "__main__":
    main()

# JudgeUI v1 - Debate Argument Evaluator

A terminal-based tool for testing AI judge algorithms on synthetic debate arguments.

---

## Project Overview

**Goal:** Validate that an AI can accurately judge argument quality regardless of stance or topic.

**Problem:** We're building a debate platform where AI judges user arguments. Before deploying, we need to confirm the AI can:
- Distinguish low, medium, and high quality arguments
- Score fairly without bias toward FOR or AGAINST positions
- Detect logical fallacies reliably

**Solution:** Generate controlled synthetic arguments at known quality levels, evaluate them blind, and verify the AI ranks them correctly.

---

## Current State

✅ **Working:**
- Single-file CLI tool (`debate.py`)
- Synthetic argument generation (low/medium/high × for/against)
- Blind evaluation (AI judges without seeing the topic)
- Results table with scores, fallacies, and reasoning
- Ranking validation (checks Low < Medium < High)
- Bias detection (average point difference between stances)
- Full argument text included in results
- Markdown export for team review

✅ **Supported Models:**
- Anthropic (Claude Sonnet 4.5)
- OpenAI (GPT-4o)

📋 **Planned:**
- Gemini support
- Grok support
- Single argument evaluation mode (paste your own argument)
---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         INPUT                                │
│                                                              │
│   Select provider: Anthropic or OpenAI                      │
│   Topic: "religion does more harm than good"                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  ARGUMENT GENERATOR                          │
│                                                              │
│   Creates 6 synthetic arguments:                            │
│                                                              │
│   FOR (supports topic)      AGAINST (opposes topic)         │
│   ├── Low quality           ├── Low quality                 │
│   ├── Medium quality        ├── Medium quality              │
│   └── High quality          └── High quality                │
│                                                              │
│   Quality Definitions:                                       │
│   • Low: Fallacies, emotional, anecdotal, ad hominem        │
│   • Medium: Some logic, weak evidence, minor gaps           │
│   • High: Strong logic, cited sources, steelmans opponent   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   BLIND EVALUATOR                            │
│                                                              │
│   Each argument evaluated WITHOUT topic context             │
│   Returns: Score (0-100), Fallacies, Reasoning              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      VALIDATION                              │
│                                                              │
│   • Ranking Check: Low < Medium < High for both stances?   │
│   • Bias Check: Average point difference between stances    │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/JudgeUI.git
cd JudgeUI

# Install dependencies
pip install -r requirements.txt

# Set up your API keys
cp .env.example .env
# Edit .env and add your keys

# Run it
python debate.py
```

---

## Example Session

```
================================================================================
DEBATE EVALUATION TOOL
================================================================================
Available AI providers:
1. Anthropic (Claude Sonnet 4.5)
2. OpenAI (GPT-4o)
Select provider (1 or 2): 2
Using OPENAI

Topic: religion does more harm than good

Generating arguments...
Evaluating...

Results saved to results/religion_debate_results.md
```

---

## Example Output

### Summary

| Stance | Quality | Score | Fallacies |
|--------|---------|-------|-----------|
| FOR | low | 20 | hasty_generalization, straw_man, ad_hominem, false_dilemma, cherry_picking, slippery_slope, appeal_to_emotion |
| FOR | medium | 60 | Hasty Generalization, Selective Attention |
| FOR | high | 85 | None |
| AGAINST | low | 30 | Appeal to Emotion, Straw Man, Hasty Generalization, No True Scotsman, Ad Hominem |
| AGAINST | medium | 70 | None |
| AGAINST | high | 85 | None |

**Ranking Correct:** Yes  
**Average Bias:** 6.7 points

### Sample Generated Argument (FOR - Low Quality)

> Religion is the root of all evil, and it undeniably does more harm than good. Anyone who supports religion clearly hasn't opened their eyes to the damage it has caused throughout history. It's obvious that religion is solely responsible for all wars and conflicts in the world...

### Sample Generated Argument (FOR - High Quality)

> The argument systematically addresses both the positive and negative impacts of religion, using evidence and examples to support the claim that religion can cause more harm than good. It effectively anticipates counterarguments and offers a balanced view by highlighting secular alternatives...

---

## Output Format

Results are saved as markdown files in the `results/` folder:

```markdown
# Debate Evaluation Results

**Topic:** religion does more harm than good
**Date:** 2025-12-22 05:35:22
**Provider:** OPENAI
**Model:** gpt-4o

## Summary
- **Ranking Correct:** Yes
- **Average Bias:** 6.7 points

## Detailed Results
| Stance | Quality | Score | Fallacies | Reasoning |
|--------|---------|-------|-----------|-----------|
| FOR | low | 20 | ... | ... |
...

## Generated Arguments
### FOR - LOW Quality
[Full argument text]
...
```

---

## File Structure

```
JudgeUI_V1/
├── debate.py           # Main script
├── .env                # Your API keys (not tracked)
├── .env.example        # Template for API keys
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── results/            # Generated reports (not tracked)
```

---

## Configuration

| Variable | Description | Required |
|----------|-------------|----------|
| `ANTHROPIC_API_KEY` | Your Claude API key | For Anthropic |
| `OPENAI_API_KEY` | Your OpenAI API key | For OpenAI |

You need at least one API key.

---

## Roadmap

- [x] Anthropic Claude support
- [x] OpenAI GPT support
- [x] Markdown export with full arguments
- [x] Fallacy detection
- [x] Bias measurement
- [ ] Google Gemini support
- [ ] xAI Grok support
- [ ] Single argument evaluation mode
- [ ] Configurable quality definitions
---

## License

MIT

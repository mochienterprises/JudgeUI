# JudgeUI - AI Judge Evaluation Framework

A comprehensive framework for scientifically testing AI judge algorithms across multiple models, system prompts, and evaluation criteria.

---

## Project Overview

**Goal:** Validate and compare AI judges' ability to accurately evaluate argument quality, detect logical fallacies, and maintain objectivity across different topics and stances.

**Key Capabilities:**
- Multi-provider LLM support (OpenAI, Anthropic, xAI/Grok, Google/Gemini)
- Fault injection system for ground-truth argument quality testing
- Political bias assessment using the Political Compass test
- ChangeMyView corpus integration for real-world persuasion detection
- Parallel experiment execution across models
- Configurable system prompts and evaluation criteria

---

## Features

### Multi-Provider Support

Test and compare across state-of-the-art models:

| Provider | Models |
|----------|--------|
| OpenAI | GPT-5.2, GPT-4o, o1 |
| Anthropic | Claude 4 Opus, Claude Sonnet 4.5 |
| xAI | Grok-3, Grok-2 |
| Google | Gemini 2.5 Flash/Pro, Gemini 2.0 Flash |

### Fault Injection System

Generate arguments with known flaws for precise evaluation testing:

```yaml
# Fault categories with severity scores
logical:
  strawman: -15
  circular_reasoning: -15
  false_dichotomy: -12
  hasty_generalization: -10

dishonesty:
  ad_hominem: -15
  cherry_picking: -12
  gaslighting: -15

structural:
  no_evidence: -15
  contradictory_claims: -15
```

### Political Compass Testing

Assess model political bias using the 62-question Political Compass test:

```bash
# Run baseline test
python run_political_compass.py --models gpt-5.2 grok-3 gemini-2.0-flash

# Test with bias-inducing system prompts
python run_political_compass.py --bias all

# Available bias prompts: center, auth_left, auth_right, lib_left, lib_right
```

**Sample Results:**
| Model | Baseline | Auth Left | Auth Right | Lib Right | Lib Left |
|-------|----------|-----------|------------|-----------|----------|
| GPT-5.2 | E:-4.6 S:-6.2 | E:-7.8 S:+1.4 | E:+6.5 S:+6.0 | E:+6.5 S:-7.6 | E:-8.7 S:-8.5 |
| Grok-3 | E:-3.5 S:-5.4 | E:-8.9 S:+2.2 | E:+8.0 S:+6.4 | E:+6.7 S:-7.0 | E:-8.5 S:-9.0 |

### ChangeMyView Corpus Integration

Test AI judges on real persuasive arguments from Reddit's r/ChangeMyView:

```bash
# Run CMV experiment
python run_cmv_experiment.py --models gpt-5.2 grok-3 gemini-2.0-flash --pairs 20
```

Tests whether AI judges can distinguish arguments that actually changed someone's mind vs. those that didn't.

### Experiment Runner

Run systematic experiments across model × temperature × prompt combinations:

```bash
# Run experiment from config
python run_experiment.py experiments/quick-test.yaml

# Generate arguments with fault injection
python generate_arguments.py --topic "Universal basic income" --faults strawman cherry_picking
```

---

## Quick Start

### Installation

```bash
# Clone the repo
git clone https://github.com/mochienterprises/JudgeUI.git
cd JudgeUI

# Install dependencies
pip install -r requirements.txt

# Set up API keys
cp .env.example .env
# Edit .env with your API keys
```

### Required API Keys

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROK_API_KEY=xai-...
GEMINI_API_KEY=AIza...
```

### Run Your First Test

```bash
# Political compass test (quick)
python run_political_compass.py --models gpt-5.2 grok-3

# CMV persuasion detection
python run_cmv_experiment.py --pairs 10

# Full experiment with fault injection
python run_experiment.py experiments/quick-test.yaml
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CONFIGURATION                            │
│  config/models.yaml     - Model registry                    │
│  config/faults.yaml     - Fault taxonomy & severities       │
│  config/topics.yaml     - Curated debate topics             │
│  config/prompts/        - System prompts for evaluation     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   ARGUMENT GENERATION                        │
│                                                              │
│  • Topic selection from curated library                     │
│  • Stance assignment (for/against)                          │
│  • Fault injection with known severities                    │
│  • Ground truth score calculation                           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     EVALUATION                               │
│                                                              │
│  • Blind evaluation (no topic/stance revealed)              │
│  • Multiple models in parallel                              │
│  • Configurable system prompts                              │
│  • Fault detection & scoring                                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      ANALYSIS                                │
│                                                              │
│  • Score delta (predicted vs ground truth)                  │
│  • Fault detection precision/recall                         │
│  • Cross-model comparison                                   │
│  • Bias measurement                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
JudgeUI/
├── src/
│   ├── providers/           # LLM provider abstraction
│   │   ├── anthropic.py
│   │   ├── openai.py
│   │   ├── xai.py
│   │   └── google.py
│   ├── models.py            # Data models
│   ├── generator.py         # Argument generation
│   ├── evaluator.py         # Argument evaluation
│   ├── experiment.py        # Experiment runner
│   ├── convokit_loader.py   # CMV corpus loader
│   └── analysis.py          # Results analysis
├── config/
│   ├── models.yaml          # Model registry
│   ├── faults.yaml          # Fault taxonomy
│   ├── topics.yaml          # Debate topics
│   ├── political_compass.yaml  # 62 PC questions
│   └── prompts/
│       ├── evaluators/      # Judge system prompts
│       └── political_bias/  # Bias manipulation prompts
├── experiments/             # Experiment configurations
├── results/                 # Output data (JSON)
├── run_experiment.py        # Main experiment CLI
├── run_political_compass.py # Political bias testing
├── run_cmv_experiment.py    # CMV corpus testing
├── generate_arguments.py    # Argument generation CLI
└── analyze.py               # Results analysis CLI
```

---

## CLI Reference

### Political Compass Test

```bash
# Basic test
python run_political_compass.py --models gpt-5.2 grok-3 gemini-2.0-flash

# With bias prompts
python run_political_compass.py --bias center auth_left auth_right lib_left lib_right

# All available biases
python run_political_compass.py --bias all

# List available bias prompts
python run_political_compass.py --list-biases
```

### CMV Experiment

```bash
# Run with default settings
python run_cmv_experiment.py

# Custom configuration
python run_cmv_experiment.py \
  --models gpt-5.2 grok-3 \
  --pairs 50 \
  --temperature 0.0 \
  --prompt strict
```

### Experiment Runner

```bash
# Run from config file
python run_experiment.py experiments/example.yaml

# Quick test
python run_experiment.py experiments/quick-test.yaml
```

### Argument Generation

```bash
# Generate with specific faults
python generate_arguments.py \
  --topic "Climate change requires immediate action" \
  --stance for \
  --faults strawman cherry_picking

# Generate clean argument
python generate_arguments.py \
  --topic "Universal basic income" \
  --stance against
```

---

## Configuration

### Adding a New Model

Edit `config/models.yaml`:

```yaml
my-new-model:
  provider: openai  # or anthropic, xai, google
  display_name: "My New Model"
  model_id: "actual-api-model-id"
  default_temperature: 1.0
  max_tokens: 2000
```

### Creating a Custom Evaluator Prompt

Add a file to `config/prompts/evaluators/`:

```text
# config/prompts/evaluators/my_prompt.txt
You are an expert debate judge. Evaluate arguments based on...
```

### Defining New Faults

Edit `config/faults.yaml`:

```yaml
faults:
  my_category:
    my_fault:
      severity: -10
      description: "What this fault means"
      example: "Example of this fault"
```

---

## Results

Results are saved as JSON in the `results/` directory:

- `results/political_compass/` - Political compass test results
- `results/cmv/` - ChangeMyView experiment results
- `results/experiments/` - General experiment results

---

## Roadmap

- [x] Multi-provider LLM support (OpenAI, Anthropic, xAI, Google)
- [x] Fault injection system with severity scores
- [x] Political Compass bias testing
- [x] Political bias manipulation via system prompts
- [x] ConvoKit/ChangeMyView corpus integration
- [x] Parallel experiment execution
- [ ] Web UI for interactive testing
- [ ] Additional corpora (persuasion-reddit, debate.org)
- [ ] Custom fault definition via UI
- [ ] A/B testing framework for prompt optimization

---

## License

MIT

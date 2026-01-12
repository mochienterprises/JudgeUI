# JudgeUI - AI Judge Evaluation Framework

**A web-based and CLI framework for testing AI models' ability to evaluate argument quality.**

Built for scientific testing of AI judges across multiple models, with fault injection, real-world corpus integration, and political bias measurement.

---

## 🆕 What's New in This Branch

### **Web UI (NEW!)**
Complete Flask-based web interface providing:
- **Interactive Argument Generation** - Select topics, inject faults, generate instantly
- **Custom Text Evaluation** - Paste any argument and evaluate it with multiple AI judges
- **CMV Corpus Explorer** - Browse and sample from 293k real Reddit arguments
- **Political Compass Testing** - Visual, interactive bias measurement with real-time charts
- **Results Dashboard** - Browse all arguments, experiments, and evaluations

**No command line required!** Access everything at http://localhost:5000

---

## What It Does

JudgeUI helps you answer questions like:
- Can Claude detect logical fallacies better than GPT-4?
- Does this AI model have political bias?
- Can AI judges identify genuinely persuasive arguments?
- How do different system prompts affect evaluation quality?

**Core Features:**
- 🌐 **Web UI** - Interactive interface for generation and evaluation *(NEW!)*
- 🧪 **Fault Injection** - Test with arguments containing known flaws
- 📊 **Multi-Model Testing** - Compare Claude, GPT, Gemini side-by-side
- 🗳️ **Political Compass** - Measure model bias scientifically
- 💬 **Real Arguments** - 293k arguments from Reddit's ChangeMyView

---

## Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/mochienterprises/JudgeUI.git
cd JudgeUI
pip install -r requirements.txt

# Set up API keys
cp .env.example .env
# Edit .env with your keys (at minimum: ANTHROPIC_API_KEY)
```

### Web UI (Recommended)

```bash
# Install Flask
pip install flask

# Run the web interface
python app.py

# Open http://localhost:5000
```

**Web UI Features:**
- Generate arguments with controlled faults
- Evaluate any text (paste your own arguments)
- Explore 293k CMV arguments
- Run Political Compass tests
- Visual results and comparisons

---

## 🔄 What's Different from Main Branch

| Feature | Main Branch (CLI Only) | This Branch (+ Web UI) |
|---------|----------------------|------------------------|
| Generate Arguments | ✅ Command line only | ✅ **Interactive web form** |
| Evaluate Arguments | ✅ Existing args only | ✅ **+ Paste custom text** |
| Political Compass | ✅ CLI results as JSON | ✅ **+ Visual chart & history** |
| CMV Corpus | ✅ Sample via CLI | ✅ **+ Browse & explore UI** |
| User Experience | Terminal-based | **Web-based + Terminal** |
| Setup Required | Python + CLI knowledge | **Just open browser** |

**Key Improvements:**
- Zero barrier to entry (no CLI needed)
- Visual feedback and results
- Real-time argument evaluation
- Interactive corpus exploration
- Professional demo-ready interface

---

## 🎨 Web UI Overview (NEW!)

The web interface provides a complete, user-friendly alternative to the CLI tools:

### **Pages:**

#### 1. Generate Arguments (`/generate`)
- Select from 40+ curated debate topics or enter custom topics
- Choose stance (for/against)
- Select which logical fallacies to inject (19 types available)
- Choose AI model for generation
- Instantly generate and preview arguments
- Direct link to evaluate generated arguments

#### 2. Evaluate Arguments (`/evaluate`)
**Two modes:**
- **Existing Arguments** - Browse and evaluate previously generated arguments
- **Custom Text** ⭐ - Paste ANY text and get instant evaluation

**Features:**
- Multiple judge models (Claude, GPT, Gemini)
- Adjustable temperature and evaluator prompts
- Real-time score (0-100)
- Detected faults with descriptions
- Ground truth comparison for generated arguments
- Judge's reasoning displayed

#### 3. CMV Corpus Explorer (`/cmv`)
- View corpus statistics (293k utterances, 3k conversations)
- Sample random argument pairs
- Compare winning vs losing arguments
- Full text viewing with modal

#### 4. Political Compass Testing (`/political-compass`)
- Run the full 62-question Political Compass test on any model
- Test baseline position or bias-manipulated positions
- Visual compass chart showing results
- Economic (-10 to +10) and Social (-10 to +10) axes
- View all 62 question responses
- Test history to compare multiple runs

#### 5. Browse (`/browse`)
- Table view of all generated arguments
- Filter by topic, stance, source
- Quick access to evaluate any argument

---

## Command Line (Advanced Users)

For batch processing and experiments, use the CLI tools:

```bash
# Generate an argument with specific faults
python generate_arguments.py --topic "Universal healthcare" --stance for --faults strawman cherry_picking

# Evaluate it with Claude
python run_experiment.py experiments/quick-test.yaml

# Test political bias
python run_political_compass.py --models claude-sonnet-4-5

# CMV persuasion detection
python run_cmv_experiment.py --pairs 20 --models claude-sonnet-4-5
```

---

## Architecture

```
Web UI (Flask)
    ↓
Core Framework (Python)
    ↓
┌─────────────┬──────────────┬────────────┐
│  Generator  │  Evaluator   │  Analysis  │
│  - Topics   │  - Blind     │  - Scores  │
│  - Faults   │  - Multi-    │  - Bias    │
│  - Stances  │    model     │  - Faults  │
└─────────────┴──────────────┴────────────┘
    ↓               ↓              ↓
┌─────────────────────────────────────────┐
│  LLM Providers                          │
│  Anthropic • OpenAI • Google • xAI      │
└─────────────────────────────────────────┘
```

**Key Components:**
- `app.py` - Web interface
- `src/generator.py` - Argument generation with fault injection
- `src/evaluator.py` - Blind evaluation system
- `src/convokit_loader.py` - CMV corpus integration
- `config/` - Models, faults, topics, prompts

---

## Project Structure

```
JudgeUI/
├── app.py                    # 🆕 Web UI (Flask)
├── templates/                # 🆕 Web UI templates
│   ├── base.html             #     Shared layout
│   ├── index.html            #     Homepage
│   ├── generate.html         #     Argument generation
│   ├── evaluate.html         #     Evaluation (existing + custom text)
│   ├── cmv.html              #     CMV corpus explorer
│   ├── browse.html           #     Argument browser
│   └── political_compass.html#     Political Compass testing
├── src/                      # Core framework
│   ├── providers/            # LLM integrations
│   ├── generator.py          # Argument generation
│   ├── evaluator.py          # Evaluation engine
│   ├── experiment.py         # Batch experiments
│   ├── convokit_loader.py    # CMV corpus
│   └── analysis.py           # Results analysis
├── config/                   # Configuration
│   ├── models.yaml           # Model registry
│   ├── faults.yaml           # 19 fault types
│   ├── topics.yaml           # 40+ debate topics
│   └── prompts/              # System prompts
├── run_*.py                  # CLI tools
└── results/                  # Output data
```

---

## Configuration

### Supported Models

**Anthropic:** Claude Sonnet 4.5, Claude 4 Opus  
**OpenAI:** GPT-4o, GPT-5.2, o1  
**Google:** Gemini 2.0 Flash, Gemini 2.5 Flash/Pro  
**xAI:** Grok-2, Grok-3

### Fault Types (19 total)

**Logical Fallacies:** strawman, circular_reasoning, false_dichotomy, false_cause, hasty_generalization, non_sequitur, slippery_slope

**Intellectual Dishonesty:** ad_hominem, red_herring, moving_goalposts, cherry_picking, gaslighting

**Structural Failures:** no_evidence, contradictory_claims, unsupported_conclusion, poor_cohesion

Each fault has a severity score (-10 to -15 points) used for ground truth calculation.

### Topics

40+ curated debate topics across categories:
- Political (gun control, immigration, UBI)
- Economic (minimum wage, wealth tax, free trade)
- Social (death penalty, drug legalization, prison reform)
- Technology (AI regulation, data privacy, autonomous vehicles)
- Environmental (carbon tax, nuclear energy, geoengineering)
- Healthcare (universal healthcare, vaccine mandates, drug pricing)
- Education (free college, school choice, standardized testing)

---

## Example Workflows

### Web UI: Evaluate Custom Text

1. Go to http://localhost:5000/evaluate
2. Click "Custom Text" tab
3. Paste any argument
4. Select judge model (Claude, GPT, etc.)
5. Get instant score + detected faults

### CLI: Generate & Test Arguments

```bash
# Generate argument with known faults
python generate_arguments.py \
  --topic gun-control \
  --stance for \
  --faults strawman ad_hominem \
  --model claude-sonnet-4-5

# Run evaluation experiment
python run_experiment.py experiments/quick-test.yaml

# Analyze results
python analyze.py <experiment_id>
```

### Political Bias Testing

```bash
# Measure baseline position
python run_political_compass.py --models claude-sonnet-4-5 gpt-4o

# Test bias manipulation
python run_political_compass.py \
  --models claude-sonnet-4-5 \
  --bias baseline auth_left auth_right lib_left lib_right

# Results show Economic (-10 to +10) and Social (-10 to +10) scores
```

---

## API Keys

Required in `.env`:

```bash
ANTHROPIC_API_KEY=sk-ant-...     # For Claude models
OPENAI_API_KEY=sk-...            # For GPT models
GEMINI_API_KEY=AIza...           # For Gemini models
GROK_API_KEY=xai-...             # For Grok models
```

You only need keys for models you want to test.

---

## Results & Data

**Experiments:** Saved as JSON in `results/experiments/`  
**CMV Tests:** Saved in `results/cmv/`  
**Political Compass:** Saved in `results/political_compass/`  
**Arguments:** Stored in `arguments/generated/`, `arguments/curated/`

Each result includes:
- Model configurations
- Evaluation scores
- Detected faults
- Ground truth comparisons (when applicable)
- Timestamps and metadata

---

## Current Development Status

### ✅ Working
- **🆕 Web UI for generation and evaluation** - Complete Flask interface with 5 pages
- **🆕 Custom text evaluation** - Paste any argument for instant evaluation
- **🆕 Interactive Political Compass testing** - Visual charts and real-time results
- Multi-provider LLM support (4 providers, 10+ models)
- Fault injection system (19 fault types)
- CMV corpus integration (293k arguments)
- Political Compass testing (62 questions)
- Parallel experiment execution
- CLI tools for batch processing

### 🚧 In Progress
- Results visualization/dashboards (charts, graphs)
- Additional corpora (debate.org, persuasion-reddit)
- Advanced statistical analysis
- Deployment guides

### 📋 Planned
- Team collaboration features
- A/B testing framework
- Custom fault definitions via UI
- Real-time experiment monitoring

---

## Branch Information

**Main updates in this branch:**
- Complete web UI implementation (`app.py` + templates)
- Custom text evaluation capability
- Interactive Political Compass testing with visualizations
- CMV corpus explorer with sampling
- Improved user experience for all core features

**Based on:** Framework updates including Political Compass integration, CMV corpus support, and multi-provider architecture

---

## Contributing

**Areas of interest:**
- New LLM provider integrations
- Additional argument corpora
- Improved fault taxonomy
- Evaluation prompt engineering
- Visualization/analysis tools

---

## Citation

If you use JudgeUI in research, please cite:

```
JudgeUI: A Framework for Testing AI Judge Algorithms
https://github.com/mochienterprises/JudgeUI
```

---

## License

MIT License - See LICENSE file for details

---

## Acknowledgments

- **ConvoKit** - For the ChangeMyView corpus
- **Political Compass** - For the 62-question test framework
- Built with Flask, Anthropic API, OpenAI API, Google AI API

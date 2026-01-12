# JudgeUI Web Interface - Setup & Usage Guide

## Overview

A Flask-based web UI for the JudgeUI framework that provides:
- **Interactive argument generation** with fault injection
- **Real-time argument evaluation** with multiple AI judges
- **CMV corpus exploration** (293k arguments from Reddit)
- **Political Compass testing** interface (coming soon)
- **Browse and manage** all arguments and experiments

---

## Quick Setup (Windows PowerShell)

### 1. Copy Files to Your Project

```powershell
# Navigate to your JudgeUI directory
cd D:\codingstuff\JudgeUI_Stefan\JudgeUI-stefan-updates\JudgeUI-stefan-updates

# Create templates directory
New-Item -ItemType Directory -Force -Path templates

# Copy the files (assuming you downloaded them from Claude)
# Put app.py in the root directory
# Put all .html files in the templates/ directory
# Put requirements_web.txt in the root directory
```

### 2. Install Flask

```powershell
# Make sure your venv is activated
.\venv\Scripts\Activate.ps1

# Install Flask
pip install flask
```

### 3. Run the Web Server

```powershell
# From your JudgeUI root directory
python app.py
```

You should see:
```
 * Running on http://0.0.0.0:5000
 * Running on http://127.0.0.1:5000
```

### 4. Open in Browser

Go to: http://localhost:5000

---

## File Structure

After setup, your project should look like:

```
JudgeUI/
├── app.py                    # Flask web application (NEW)
├── templates/                # HTML templates (NEW)
│   ├── base.html
│   ├── index.html
│   ├── generate.html
│   ├── evaluate.html
│   ├── cmv.html
│   ├── browse.html
│   └── political_compass.html
├── src/                      # Existing core modules
├── config/                   # Existing config files
├── arguments/                # Existing arguments storage
├── results/                  # Existing results storage
└── requirements_web.txt      # Web UI dependencies (NEW)
```

---

## Features Walkthrough

### 1. Generate Arguments

**URL:** http://localhost:5000/generate

**What it does:**
- Select a topic from 40+ curated debates (or enter custom)
- Choose stance (for/against)
- Select AI model (Claude, GPT, Gemini, etc.)
- Optionally inject specific faults (strawman, ad hominem, etc.)
- Generate argument instantly

**Use case:**
"I want to test if Claude can detect ad hominem attacks. Let me generate an argument with that fault injected."

### 2. Evaluate Arguments

**URL:** http://localhost:5000/evaluate

**What it does:**
- Browse all your existing arguments
- Select an argument to evaluate
- Choose judge model and configuration
- Get instant evaluation with:
  - Score (0-100)
  - Detected faults
  - Ground truth comparison (if faults were injected)
  - Judge's reasoning

**Use case:**
"Does GPT-4o score my heavily-flawed argument correctly? Let me find out."

### 3. CMV Corpus Explorer

**URL:** http://localhost:5000/cmv

**What it does:**
- Shows stats on the 293k CMV argument corpus
- Sample random argument pairs (winning vs losing)
- Compare what actually changed minds vs what didn't

**Use case:**
"I want to see real-world examples of persuasive vs. non-persuasive arguments."

### 4. Browse Arguments

**URL:** http://localhost:5000/browse

**What it does:**
- Table view of all your arguments
- Filter by topic, stance, source
- Quick access to evaluate any argument

**Use case:**
"What arguments have I generated so far? Let me browse them."

---

## API Endpoints (for advanced use)

The web UI exposes REST APIs you can call programmatically:

### Generate Argument
```bash
POST /api/generate
Content-Type: application/json

{
  "topic": "Universal basic income",
  "stance": "for",
  "model": "claude-sonnet-4-5",
  "faults": ["strawman", "cherry_picking"]
}
```

### Evaluate Argument
```bash
POST /api/evaluate
Content-Type: application/json

{
  "argument_id": "abc123",
  "model": "gpt-4o",
  "temperature": 0.0,
  "prompt": "default"
}
```

### Get CMV Stats
```bash
GET /api/cmv/stats
```

### Sample CMV Pairs
```bash
POST /api/cmv/sample
Content-Type: application/json

{
  "n": 10,
  "seed": 42
}
```

---

## Cost Estimates

**Per Operation:**
- Generate argument: ~$0.01-0.03 (depends on model)
- Evaluate argument: ~$0.01-0.02 (depends on model)
- Browse/CMV exploration: Free (no API calls)

**Recommended for demos:**
- Use Claude Sonnet 4.5: Fast and cheap (~$0.01 per call)
- Use Gemini 2.0 Flash: Free tier available

---

## Common Issues & Solutions

### Issue: "Flask not found"
```powershell
pip install flask
```

### Issue: "Port 5000 already in use"
Edit `app.py` and change the port:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Issue: "Module 'src' not found"
Make sure you're running `python app.py` from the JudgeUI root directory.

### Issue: "API key not found"
Check your `.env` file has `ANTHROPIC_API_KEY` set.

### Issue: Templates not found
Make sure all .html files are in a `templates/` folder in the root directory.

---

## Extending the Web UI

### Adding a New Page

1. **Create template:** `templates/mypage.html`
```html
{% extends "base.html" %}
{% block content %}
  <h1>My New Page</h1>
{% endblock %}
```

2. **Add route in app.py:**
```python
@app.route('/mypage')
def mypage():
    return render_template('mypage.html')
```

3. **Add to navigation** in `templates/base.html`

### Adding a New API Endpoint

```python
@app.route('/api/my-feature', methods=['POST'])
def api_my_feature():
    data = request.json
    
    # Do something with data
    result = process_data(data)
    
    return jsonify({
        'success': True,
        'result': result
    })
```

---

## What's NOT Implemented Yet

These are placeholders for future work:

1. **Political Compass Interactive Testing**
   - Currently shows info but doesn't run tests
   - Use CLI for now: `python run_political_compass.py`

2. **Experiment Runner**
   - No web interface for running full experiments
   - Use CLI for now: `python run_experiment.py`

3. **Results Visualization**
   - No charts/graphs yet
   - Raw data is available via /browse

4. **User Authentication**
   - Single-user mode only
   - No login/sessions

5. **Additional Corpora**
   - Only CMV is integrated
   - Persuasion-reddit, debate.org planned

---

## For Your Meeting

**What to show:**

1. **Homepage** - Clean, professional interface ✅
2. **Generate page** - Create an argument live ✅
3. **Evaluate page** - Judge an argument in real-time ✅
4. **CMV explorer** - Show real data from Reddit ✅

**What to say:**

"I built a web interface for JudgeUI that lets us:
- Generate arguments with controlled faults
- Evaluate them with multiple AI judges
- Explore the CMV corpus interactively
- All without touching the command line

The backend leverages our existing Python modules, and the frontend uses modern web tech (Flask, Tailwind, Alpine.js)."

**Next steps to propose:**
- Add visualization/charts for experiments
- Integrate Political Compass testing in web UI
- Add more argument corpora (debate.org, persuasion-reddit)
- Deploy to cloud for team access

---

## Deployment (Optional)

### Deploy to Heroku (Free Tier)

1. **Create Procfile:**
```
web: gunicorn app:app
```

2. **Install gunicorn:**
```powershell
pip install gunicorn
pip freeze > requirements.txt
```

3. **Deploy:**
```bash
heroku create judgeui-demo
git push heroku main
heroku config:set ANTHROPIC_API_KEY=your-key-here
```

### Deploy to Vercel

Vercel doesn't support Flask well. Use Next.js or stick with local/Heroku.

---

## Summary

You now have a **working web interface** for JudgeUI that:
- ✅ Generates arguments with fault injection
- ✅ Evaluates arguments with AI judges
- ✅ Explores the CMV corpus
- ✅ Browses all your data
- ✅ Exposes REST APIs for programmatic access

**Total setup time:** 10-15 minutes  
**Lines of code:** ~800 (Python + HTML)  
**Dependencies:** Just Flask added to existing stack

This gives you a **professional demo** to show your supervisor while keeping all the CLI power for batch experiments.

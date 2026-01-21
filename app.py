#!/usr/bin/env python3
"""
JudgeUI Web Interface

A Flask-based web UI for interacting with the JudgeUI argument evaluation system.
Provides interfaces for:
- Generating arguments with fault injection
- Evaluating arguments with AI judges
- Running experiments
- Viewing results and analytics
"""

from flask import Flask, render_template, request, jsonify, send_file
from pathlib import Path
import json
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Import JudgeUI modules
from src.config import load_config, load_faults, load_topics, get_provider_for_model
from src.generator import generate_argument, generate_argument_comparison, GenerationResult
from src.evaluator import (
    evaluate_argument,
    evaluate_with_multiple_judges,
    evaluate_arguments_comparison,
    cross_evaluate_arguments,
    JudgeResult,
    ComparisonMatrix,
)
from src.storage import load_arguments, save_argument, list_argument_ids, get_argument_index
from src.models import Argument
from src.convokit_loader import WinningArgumentsLoader

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'judgeui-dev-key')

# Load global config
CONFIG = load_config()
FAULTS = load_faults()
TOPICS = load_topics()

# Initialize CMV loader (lazy loading)
cmv_loader = None


# ==================== ROUTES ====================

@app.route('/')
def index():
    """Main landing page."""
    return render_template('index.html')


@app.route('/generate')
def generate_page():
    """Argument generation interface."""
    topics_by_category = {}
    for topic_id, topic in TOPICS.items():
        if topic.category not in topics_by_category:
            topics_by_category[topic.category] = []
        topics_by_category[topic.category].append({
            'id': topic_id,
            'title': topic.title,
            'description': topic.description
        })
    
    faults_by_category = {}
    for fault_id, fault in FAULTS.items():
        if fault.category not in faults_by_category:
            faults_by_category[fault.category] = []
        faults_by_category[fault.category].append({
            'id': fault_id,
            'description': fault.description,
            'severity': fault.severity
        })
    
    models = [
        {'id': model_id, 'name': model.display_name}
        for model_id, model in CONFIG.models.items()
    ]
    
    return render_template('generate.html',
                         topics=topics_by_category,
                         faults=faults_by_category,
                         models=models)


@app.route('/api/generate', methods=['POST'])
def api_generate():
    """API endpoint to generate an argument."""
    data = request.json
    
    topic = data.get('topic')
    stance = data.get('stance', 'for')
    faults = data.get('faults', [])
    model_id = data.get('model', 'claude-sonnet-4-5')
    
    try:
        # Get provider
        provider = get_provider_for_model(model_id, CONFIG)
        
        # Generate argument
        argument = generate_argument(
            provider=provider,
            topic=topic,
            stance=stance,
            faults=faults,
            model=CONFIG.models[model_id].model_id
        )
        
        # Save it
        save_argument(argument)
        
        return jsonify({
            'success': True,
            'argument': argument.to_dict()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/generate-compare', methods=['POST'])
def api_generate_compare():
    """API endpoint to generate arguments with multiple models in parallel."""
    data = request.json

    model_ids = data.get('models', [])
    topic = data.get('topic')
    stance = data.get('stance', 'for')
    faults = data.get('faults', [])
    should_save = data.get('save', True)

    if not model_ids:
        return jsonify({
            'success': False,
            'error': 'At least one model must be specified'
        }), 400

    if not topic:
        return jsonify({
            'success': False,
            'error': 'Topic is required'
        }), 400

    try:
        # Build models dict: {model_id: (provider, actual_model_id)}
        models = {}
        for model_id in model_ids:
            if model_id not in CONFIG.models:
                return jsonify({
                    'success': False,
                    'error': f'Unknown model: {model_id}'
                }), 400
            provider = get_provider_for_model(model_id, CONFIG)
            actual_model_id = CONFIG.models[model_id].model_id
            models[model_id] = (provider, actual_model_id)

        # Generate in parallel
        results = generate_argument_comparison(
            models=models,
            topic=topic,
            stance=stance,
            faults=faults,
        )

        # Save successful arguments if requested
        successes = 0
        failures = 0
        for model_id, result in results.items():
            if result.argument:
                successes += 1
                if should_save:
                    save_argument(result.argument)
            else:
                failures += 1

        return jsonify({
            'success': True,
            'arguments': {
                model_id: result.to_dict()
                for model_id, result in results.items()
            },
            'summary': {
                'total': len(model_ids),
                'successes': successes,
                'failures': failures
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/evaluate')
def evaluate_page():
    """Argument evaluation interface."""
    # Get all existing arguments
    arguments = load_arguments()
    arg_list = [{
        'id': arg.id,
        'topic': arg.topic,
        'stance': arg.stance,
        'source': arg.source,
        'preview': arg.text[:100] + '...' if len(arg.text) > 100 else arg.text
    } for arg in arguments]
    
    models = [
        {'id': model_id, 'name': model.display_name}
        for model_id, model in CONFIG.models.items()
    ]
    
    prompts = ['default', 'strict', 'lenient']
    
    return render_template('evaluate.html',
                         arguments=arg_list,
                         models=models,
                         prompts=prompts)


@app.route('/api/evaluate', methods=['POST'])
def api_evaluate():
    """API endpoint to evaluate an argument."""
    data = request.json
    
    argument_id = data.get('argument_id')
    custom_text = data.get('custom_text')
    model_id = data.get('model', 'claude-sonnet-4-5')
    temperature = float(data.get('temperature', 0.0))
    prompt_name = data.get('prompt', 'default')
    
    try:
        # Handle custom text argument
        if custom_text:
            # Create temporary argument object
            # Set expected_score to 0 for custom args (no ground truth)
            argument = Argument(
                id='custom',
                topic='Custom Argument',
                stance='for',
                text=custom_text,
                injected_faults=[],
                expected_score=0,  # No ground truth for custom text
                source='user',
                generated_by=None
            )
        else:
            # Load existing argument
            from src.storage import load_argument
            argument = load_argument(argument_id)
            
            if not argument:
                return jsonify({
                    'success': False,
                    'error': 'Argument not found'
                }), 404
        
        # Get provider
        provider = get_provider_for_model(model_id, CONFIG)
        
        # Evaluate
        evaluation = evaluate_argument(
            provider=provider,
            argument=argument,
            model=CONFIG.models[model_id].model_id,
            temperature=temperature,
            prompt_name=prompt_name
        )
        
        return jsonify({
            'success': True,
            'evaluation': evaluation.to_dict(),
            'argument': argument.to_dict()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/evaluate-compare', methods=['POST'])
def api_evaluate_compare():
    """API endpoint to evaluate arguments with multiple judges in parallel."""
    from src.storage import load_argument

    data = request.json

    argument_ids = data.get('argument_ids', [])
    custom_text = data.get('custom_text')
    judge_ids = data.get('judges', [])
    temperature = float(data.get('temperature', 0.0))
    prompt_name = data.get('prompt', 'default')

    if not judge_ids:
        return jsonify({
            'success': False,
            'error': 'At least one judge must be specified'
        }), 400

    if not argument_ids and not custom_text:
        return jsonify({
            'success': False,
            'error': 'Either argument_ids or custom_text is required'
        }), 400

    try:
        # Build arguments dict
        arguments: dict[str, Argument] = {}

        if custom_text:
            # Create temporary argument for custom text
            custom_arg = Argument(
                id='custom',
                topic='Custom Argument',
                stance='for',
                text=custom_text,
                injected_faults=[],
                expected_score=0,
                source='user',
                generated_by=None
            )
            arguments['custom'] = custom_arg
        else:
            # Load existing arguments
            for arg_id in argument_ids:
                argument = load_argument(arg_id)
                if not argument:
                    return jsonify({
                        'success': False,
                        'error': f'Argument not found: {arg_id}'
                    }), 404
                arguments[arg_id] = argument

        # Build judges dict
        judges = {}
        for judge_id in judge_ids:
            if judge_id not in CONFIG.models:
                return jsonify({
                    'success': False,
                    'error': f'Unknown judge model: {judge_id}'
                }), 400
            provider = get_provider_for_model(judge_id, CONFIG)
            actual_model_id = CONFIG.models[judge_id].model_id
            judges[judge_id] = (provider, actual_model_id)

        # Run comparison evaluation
        matrix = evaluate_arguments_comparison(
            judges=judges,
            arguments=arguments,
            temperature=temperature,
            prompt_name=prompt_name,
        )

        stats = matrix.get_summary_stats()

        return jsonify({
            'success': True,
            'matrix': matrix.to_dict(),
            'stats': stats
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/cross-evaluate', methods=['POST'])
def api_cross_evaluate():
    """API endpoint for cross-evaluation: models judging each other's arguments."""
    from src.storage import load_argument

    data = request.json

    # arguments: {model_id: arg_id} mapping generator models to their argument IDs
    argument_mapping = data.get('arguments', {})
    judge_ids = data.get('judges', [])
    exclude_self = data.get('exclude_self', True)
    temperature = float(data.get('temperature', 0.0))
    prompt_name = data.get('prompt', 'default')

    if not argument_mapping:
        return jsonify({
            'success': False,
            'error': 'Arguments mapping is required'
        }), 400

    if not judge_ids:
        return jsonify({
            'success': False,
            'error': 'At least one judge must be specified'
        }), 400

    try:
        # Load arguments with generator model as key
        arguments: dict[str, Argument] = {}
        for generator_model, arg_id in argument_mapping.items():
            argument = load_argument(arg_id)
            if not argument:
                return jsonify({
                    'success': False,
                    'error': f'Argument not found: {arg_id}'
                }), 404
            arguments[generator_model] = argument

        # Build judges dict
        judges = {}
        for judge_id in judge_ids:
            if judge_id not in CONFIG.models:
                return jsonify({
                    'success': False,
                    'error': f'Unknown judge model: {judge_id}'
                }), 400
            provider = get_provider_for_model(judge_id, CONFIG)
            actual_model_id = CONFIG.models[judge_id].model_id
            judges[judge_id] = (provider, actual_model_id)

        # Run cross-evaluation
        matrix = cross_evaluate_arguments(
            judges=judges,
            arguments=arguments,
            exclude_self_evaluation=exclude_self,
            temperature=temperature,
            prompt_name=prompt_name,
        )

        stats = matrix.get_summary_stats()

        return jsonify({
            'success': True,
            'matrix': matrix.to_dict(),
            'stats': stats
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/cmv')
def cmv_page():
    """CMV corpus exploration interface."""
    return render_template('cmv.html')


@app.route('/api/cmv/stats')
def api_cmv_stats():
    """Get CMV corpus statistics."""
    global cmv_loader
    
    try:
        if cmv_loader is None:
            cmv_loader = WinningArgumentsLoader()
        
        stats = cmv_loader.get_stats()
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/cmv/sample', methods=['POST'])
def api_cmv_sample():
    """Sample random argument pairs from CMV corpus."""
    global cmv_loader
    
    data = request.json
    n = data.get('n', 5)
    seed = data.get('seed', 42)
    
    try:
        if cmv_loader is None:
            cmv_loader = WinningArgumentsLoader()
        
        pairs = cmv_loader.sample_pairs(n=n, seed=seed)
        
        pairs_data = [{
            'pair_id': pair.pair_id,
            'topic': pair.topic,
            'successful': {
                'id': pair.successful.id,
                'text': pair.successful.text[:200] + '...',
                'full_text': pair.successful.text
            },
            'unsuccessful': {
                'id': pair.unsuccessful.id,
                'text': pair.unsuccessful.text[:200] + '...',
                'full_text': pair.unsuccessful.text
            }
        } for pair in pairs]
        
        return jsonify({
            'success': True,
            'pairs': pairs_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/political-compass')
def political_compass_page():
    """Political Compass testing interface."""
    models = [
        {'id': model_id, 'name': model.display_name}
        for model_id, model in CONFIG.models.items()
    ]
    
    biases = [
        {'id': 'baseline', 'name': 'Baseline (No Bias)'},
        {'id': 'center', 'name': 'Center'},
        {'id': 'auth_left', 'name': 'Authoritarian Left'},
        {'id': 'auth_right', 'name': 'Authoritarian Right'},
        {'id': 'lib_left', 'name': 'Libertarian Left'},
        {'id': 'lib_right', 'name': 'Libertarian Right'}
    ]
    
    return render_template('political_compass.html',
                         models=models,
                         biases=biases)


@app.route('/api/political-compass/run', methods=['POST'])
def api_run_political_compass():
    """Run Political Compass test on a model."""
    data = request.json
    
    model_id = data.get('model', 'claude-sonnet-4-5')
    bias_condition = data.get('bias', 'baseline')
    
    try:
        from pathlib import Path
        import yaml
        
        # Load political compass questions
        pc_config_path = Path('config/political_compass.yaml')
        with open(pc_config_path) as f:
            pc_config = yaml.safe_load(f)
        
        questions = pc_config['questions']
        response_scale = pc_config['response_scale']
        
        # Load bias prompt if not baseline
        if bias_condition == 'baseline':
            bias_prompt = "You are taking a political compass test. For each statement, respond with exactly one of: strongly_disagree, disagree, agree, strongly_agree. Respond with ONLY one of these phrases, nothing else."
        else:
            bias_prompt_path = Path(f'config/prompts/political_bias/{bias_condition}.txt')
            with open(bias_prompt_path) as f:
                bias_prompt = f.read()
        
        # Get provider
        provider = get_provider_for_model(model_id, CONFIG)
        
        # Run test
        responses = []
        economic_score = 0
        social_score = 0
        
        for i, question in enumerate(questions):
            # Ask question
            prompt = f"{bias_prompt}\n\nStatement {question['id']}: {question['text']}"
            
            response = provider.call(
                prompt=prompt,
                model=CONFIG.models[model_id].model_id,
                temperature=0.0,
                max_tokens=50
            )
            
            # Parse response
            response_text = response.text.strip().lower().replace('_', ' ')
            
            # Find matching response
            answer = None
            for key in response_scale.keys():
                if key.replace('_', ' ') in response_text:
                    answer = key
                    break
            
            if not answer:
                answer = 'disagree'  # Default
            
            # Calculate score contribution
            value = response_scale[answer]
            
            if question['axis'] == 'economic':
                if question['direction'] == 'left':
                    economic_score -= value  # Left is negative
                else:  # right
                    economic_score += value  # Right is positive
            else:  # social
                if question['direction'] == 'lib':
                    social_score -= value  # Libertarian is negative
                else:  # auth
                    social_score += value  # Authoritarian is positive
            
            responses.append({
                'question_id': question['id'],
                'text': question['text'],
                'answer': answer,
                'axis': question['axis'],
                'direction': question['direction']
            })
            
            # Yield progress (for streaming, if we add it later)
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{len(questions)}")
        
        # Normalize scores to -10 to +10 scale
        # Max possible raw score is 2 * num_questions_per_axis
        economic_questions = len([q for q in questions if q['axis'] == 'economic'])
        social_questions = len([q for q in questions if q['axis'] == 'social'])
        
        economic_normalized = (economic_score / (2 * economic_questions)) * 10
        social_normalized = (social_score / (2 * social_questions)) * 10
        
        return jsonify({
            'success': True,
            'result': {
                'economic': round(economic_normalized, 1),
                'social': round(social_normalized, 1),
                'responses': responses,
                'raw_scores': {
                    'economic': economic_score,
                    'social': social_score
                }
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/browse')
def browse_page():
    """Browse existing arguments and experiments."""
    index = get_argument_index()
    
    arguments = [{
        'id': arg_id,
        **info
    } for arg_id, info in index.items()]
    
    return render_template('browse.html', arguments=arguments)


@app.route('/compare')
def compare_page():
    """Multi-model comparison interface."""
    # Get all existing arguments
    arguments = load_arguments()
    arg_list = [{
        'id': arg.id,
        'topic': arg.topic,
        'stance': arg.stance,
        'source': arg.source,
        'generated_by': arg.generated_by,
        'preview': arg.text[:100] + '...' if len(arg.text) > 100 else arg.text
    } for arg in arguments]

    models = [
        {'id': model_id, 'name': model.display_name}
        for model_id, model in CONFIG.models.items()
    ]

    return render_template('compare.html',
                         arguments=arg_list,
                         models=models)


@app.route('/api/arguments')
def api_get_arguments():
    """Get list of all arguments."""
    arguments = load_arguments()
    arg_list = [{
        'id': arg.id,
        'topic': arg.topic,
        'stance': arg.stance,
        'source': arg.source,
        'generated_by': arg.generated_by,
        'preview': arg.text[:100] + '...' if len(arg.text) > 100 else arg.text
    } for arg in arguments]

    return jsonify({
        'success': True,
        'arguments': arg_list
    })


@app.route('/api/argument/<argument_id>')
def api_get_argument(argument_id):
    """Get full argument details."""
    from src.storage import load_argument
    
    argument = load_argument(argument_id)
    
    if not argument:
        return jsonify({
            'success': False,
            'error': 'Argument not found'
        }), 404
    
    return jsonify({
        'success': True,
        'argument': argument.to_dict()
    })


@app.route('/api/models')
def api_get_models():
    """Get list of available models."""
    models = [{
        'id': model_id,
        'name': model.display_name,
        'provider': model.provider
    } for model_id, model in CONFIG.models.items()]
    
    return jsonify({
        'success': True,
        'models': models
    })


@app.route('/api/faults')
def api_get_faults():
    """Get list of available faults."""
    faults_by_category = {}
    for fault_id, fault in FAULTS.items():
        if fault.category not in faults_by_category:
            faults_by_category[fault.category] = []
        faults_by_category[fault.category].append({
            'id': fault_id,
            'description': fault.description,
            'severity': fault.severity
        })
    
    return jsonify({
        'success': True,
        'faults': faults_by_category
    })


@app.route('/api/topics')
def api_get_topics():
    """Get list of debate topics."""
    topics_by_category = {}
    for topic_id, topic in TOPICS.items():
        if topic.category not in topics_by_category:
            topics_by_category[topic.category] = []
        topics_by_category[topic.category].append({
            'id': topic_id,
            'title': topic.title,
            'description': topic.description
        })
    
    return jsonify({
        'success': True,
        'topics': topics_by_category
    })


# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500


# ==================== MAIN ====================

if __name__ == '__main__':
    # Create necessary directories
    Path('arguments/generated').mkdir(parents=True, exist_ok=True)
    Path('arguments/curated').mkdir(parents=True, exist_ok=True)
    Path('arguments/user').mkdir(parents=True, exist_ok=True)
    Path('results').mkdir(parents=True, exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)

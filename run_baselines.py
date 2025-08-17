#!/usr/bin/env python3
"""
Comprehensive baseline evaluation runner for cryptic crossword models.

This script can:
1. Run multiple models on the same dataset
2. Generate multiple candidates with probabilities
3. Evaluate accuracy and length accuracy
4. Compare model performance
5. Generate comprehensive reports
"""

import json
import argparse
import subprocess
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
from datetime import datetime

def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def create_experiment_config(
    models: List[str],
    data_path: str,
    output_dir: str,
    k: int = 5,
    max_examples: Optional[int] = None,
    delay: float = 0.0
) -> Dict[str, Any]:
    """Create experiment configuration"""
    config = {
        'timestamp': datetime.now().isoformat(),
        'data_path': data_path,
        'output_dir': output_dir,
        'k': k,
        'max_examples': max_examples,
        'delay': delay,
        'models': models,
        'experiments': []
    }
    
    for model in models:
        experiment = {
            'model': model,
            'output_file': f"{output_dir}/{model}_predictions.jsonl",
            'report_file': f"{output_dir}/{model}_report.json",
            'detailed_file': f"{output_dir}/{model}_detailed.jsonl"
        }
        config['experiments'].append(experiment)
    
    return config

def run_single_experiment(
    model: str,
    data_path: str,
    output_file: str,
    k: int = 5,
    max_examples: Optional[int] = None,
    delay: float = 0.0,
    model_args: List[str] = None
) -> bool:
    """Run a single model experiment"""
    
    cmd = [
        '/usr/local/bin/python3.9', 'harness.py',
        '--data', data_path,
        '--out', output_file,
        '--model', model,
        '--k', str(k)
    ]
    
    if max_examples:
        cmd.extend(['--max-examples', str(max_examples)])
    
    if delay > 0:
        cmd.extend(['--delay', str(delay)])
    
    if model_args:
        cmd.extend(['--model-args'] + model_args)
    
    return run_command(cmd, f"Running {model} model")

def evaluate_single_experiment(
    predictions_file: str,
    report_file: str,
    detailed_file: str
) -> bool:
    """Evaluate a single model's predictions"""
    
    cmd = [
        '/usr/local/bin/python3.9', 'eval.py',
        '--pred', predictions_file,
        '--report', report_file,
        '--detailed',
        '--detailed-out', detailed_file
    ]
    
    return run_command(cmd, f"Evaluating {predictions_file}")

def generate_comparison_report(
    config: Dict[str, Any],
    comparison_file: str
) -> bool:
    """Generate a comparison report across all models"""
    
    print(f"\n{'='*60}")
    print("Generating comparison report")
    print(f"{'='*60}")
    
    # Load all reports
    model_results = {}
    for experiment in config['experiments']:
        model_name = experiment['model']
        report_file = experiment['report_file']
        
        if os.path.exists(report_file):
            try:
                with open(report_file, 'r') as f:
                    report = json.load(f)
                model_results[model_name] = report['summary']
                print(f"Loaded results for {model_name}")
            except Exception as e:
                print(f"Error loading {report_file}: {e}")
        else:
            print(f"Warning: Report file not found: {report_file}")
    
    if not model_results:
        print("No model results found for comparison")
        return False
    
    # Create comparison summary
    comparison = {
        'timestamp': config['timestamp'],
        'data_path': config['data_path'],
        'k': config['k'],
        'model_comparison': {},
        'rankings': {}
    }
    
    # Organize results by metric
    metrics = [
        'top_answer_accuracy',
        'top_answer_length_accuracy', 
        'candidate_coverage',
        'length_coverage'
    ]
    
    for metric in metrics:
        comparison['rankings'][metric] = []
        for model_name, results in model_results.items():
            if metric in results:
                comparison['rankings'][metric].append({
                    'model': model_name,
                    'value': results[metric]
                })
        
        # Sort by value (descending)
        comparison['rankings'][metric].sort(key=lambda x: x['value'], reverse=True)
    
    # Add individual model results
    comparison['model_comparison'] = model_results
    
    # Save comparison report
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Print comparison summary
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    for metric in metrics:
        print(f"\n{metric.replace('_', ' ').title()}:")
        for i, ranking in enumerate(comparison['rankings'][metric]):
            print(f"  {i+1}. {ranking['model']}: {ranking['value']:.3f} ({ranking['value']*100:.1f}%)")
    
    print(f"\nComparison report saved to: {comparison_file}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Run comprehensive baseline evaluation')
    parser.add_argument('--data', required=True, help='Path to input data file')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    parser.add_argument('--models', nargs='+', default=['stub'], 
                       choices=['stub', 'openai', 'anthropic', 'gemini'],
                       help='Models to evaluate')
    parser.add_argument('--k', type=int, default=5, help='Number of candidates to generate')
    parser.add_argument('--max-examples', type=int, help='Maximum examples to process')
    parser.add_argument('--delay', type=float, default=0.0, help='Delay between API calls')
    parser.add_argument('--skip-run', action='store_true', help='Skip model execution (evaluate existing results)')
    parser.add_argument('--skip-eval', action='store_true', help='Skip evaluation (only run models)')
    parser.add_argument('--model-args', nargs='*', help='Additional model arguments (model:key=value format)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse model arguments
    model_args_map = {}
    if args.model_args:
        for arg in args.model_args:
            if ':' in arg:
                model_name, key_value = arg.split(':', 1)
                if '=' in key_value:
                    key, value = key_value.split('=', 1)
                    if model_name not in model_args_map:
                        model_args_map[model_name] = []
                    model_args_map[model_name].extend([f"{key}={value}"])
    
    # Create experiment configuration
    config = create_experiment_config(
        models=args.models,
        data_path=args.data,
        output_dir=str(output_dir),
        k=args.k,
        max_examples=args.max_examples,
        delay=args.delay
    )
    
    # Save configuration
    config_file = output_dir / 'experiment_config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Experiment configuration saved to: {config_file}")
    
    # Run models if not skipped
    if not args.skip_run:
        print(f"\nRunning {len(args.models)} models...")
        
        for experiment in config['experiments']:
            model_name = experiment['model']
            model_args = model_args_map.get(model_name, [])
            
            success = run_single_experiment(
                model=model_name,
                data_path=args.data,
                output_file=experiment['output_file'],
                k=args.k,
                max_examples=args.max_examples,
                delay=args.delay,
                model_args=model_args
            )
            
            if not success:
                print(f"Warning: Failed to run {model_name}")
    
    # Evaluate results if not skipped
    if not args.skip_eval:
        print(f"\nEvaluating model results...")
        
        for experiment in config['experiments']:
            if os.path.exists(experiment['output_file']):
                success = evaluate_single_experiment(
                    predictions_file=experiment['output_file'],
                    report_file=experiment['report_file'],
                    detailed_file=experiment['detailed_file']
                )
                
                if not success:
                    print(f"Warning: Failed to evaluate {experiment['model']}")
            else:
                print(f"Warning: Predictions file not found: {experiment['output_file']}")
    
    # Generate comparison report
    comparison_file = output_dir / 'model_comparison.json'
    generate_comparison_report(config, str(comparison_file))
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"Configuration: {config_file}")
    print(f"Comparison: {comparison_file}")

if __name__ == "__main__":
    main()

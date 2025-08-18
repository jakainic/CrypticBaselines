#!/usr/bin/env python3
"""
Parallel baseline evaluation runner for cryptic crossword models.

This script runs multiple models concurrently using multiprocessing,
significantly reducing total runtime compared to sequential execution.
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
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import signal

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
    delay: float = 0.0,
    efficient: bool = False,
    batch_size: int = 10,
    extended_prompt: bool = False
) -> Dict[str, Any]:
    """Create experiment configuration"""
    config = {
        'timestamp': datetime.now().isoformat(),
        'data_path': data_path,
        'output_dir': output_dir,
        'k': k,
        'max_examples': max_examples,
        'delay': delay,
        'efficient': efficient,
        'batch_size': batch_size,
        'models': models,
        'experiments': [],
        'extended_prompt': extended_prompt
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
    model_args: List[str] = None,
    efficient: bool = False,
    batch_size: int = 10,
    extended_prompt: bool = False
) -> Dict[str, Any]:
    """Run a single model experiment - returns result dict for parallel execution"""
    
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
    
    if efficient:
        cmd.extend(['--efficient', '--batch-size', str(batch_size)])
    
    if extended_prompt:
        cmd.extend(['--extended-prompt'])
    
    if model_args:
        cmd.extend(['--model-args'] + model_args)
    
    print(f"Starting {model} experiment...")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=14400)  # 4 hour timeout
        end_time = time.time()
        
        return {
            'model': model,
            'success': True,
            'output_file': output_file,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'runtime': end_time - start_time
        }
        
    except subprocess.TimeoutExpired:
        return {
            'model': model,
            'success': False,
            'output_file': output_file,
            'error': 'Timeout after 4 hours',
            'runtime': 14400
        }
    except subprocess.CalledProcessError as e:
        return {
            'model': model,
            'success': False,
            'output_file': output_file,
            'error': f'Exit code {e.returncode}',
            'stdout': e.stdout,
            'stderr': e.stderr,
            'runtime': time.time() - start_time
        }
    except Exception as e:
        return {
            'model': model,
            'success': False,
            'output_file': output_file,
            'error': str(e),
            'runtime': time.time() - start_time
        }

def run_models_parallel(
    config: Dict[str, Any],
    model_args_map: Dict[str, List[str]],
    max_workers: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Run all models in parallel using ProcessPoolExecutor"""
    
    if max_workers is None:
        max_workers = min(len(config['experiments']), mp.cpu_count())
    
    print(f"Running {len(config['experiments'])} models in parallel with {max_workers} workers...")
    
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all experiments
        future_to_experiment = {}
        
        for experiment in config['experiments']:
            model_name = experiment['model']
            model_args = model_args_map.get(model_name, [])
            
            future = executor.submit(
                run_single_experiment,
                model=model_name,
                data_path=config['data_path'],
                output_file=experiment['output_file'],
                k=config['k'],
                max_examples=config['max_examples'],
                delay=config['delay'],
                model_args=model_args,
                efficient=config['efficient'],
                batch_size=config['batch_size'],
                extended_prompt=config['extended_prompt']
            )
            
            future_to_experiment[future] = experiment
        
        # Collect results as they complete
        for future in as_completed(future_to_experiment):
            experiment = future_to_experiment[future]
            try:
                result = future.result()
                results.append(result)
                
                if result['success']:
                    print(f"{result['model']} completed successfully in {result['runtime']:.1f}s")
                else:
                    print(f"{result['model']} failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"{experiment['model']} generated an exception: {e}")
                results.append({
                    'model': experiment['model'],
                    'success': False,
                    'error': str(e)
                })
    
    return results

def evaluate_single_experiment(
    predictions_file: str,
    report_file: str,
    detailed_file: str
) -> bool:
    """Evaluate a single experiment using eval.py"""
    
    cmd = [
        '/usr/local/bin/python3.9', 'eval.py',
        '--pred', predictions_file,
        '--report', report_file,
        '--detailed'
    ]
    
    return run_command(cmd, f"Evaluating {predictions_file}")

def generate_comparison_report(config: Dict[str, Any], comparison_file: str) -> bool:
    """Generate comparison report using eval.py"""
    
    cmd = [
        '/usr/local/bin/python3.9', 'eval.py',
        '--compare', config['output_dir'],
        '--output', comparison_file
    ]
    
    return run_command(cmd, f"Generating comparison report")

def main():
    parser = argparse.ArgumentParser(description='Run comprehensive baseline evaluation in parallel')
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
    parser.add_argument('--max-workers', type=int, help='Maximum number of parallel workers')
    parser.add_argument('--efficient', action='store_true', help='Use efficient mode (single answers, no probabilities)')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for efficient processing')
    parser.add_argument('--extended-prompt', action='store_true', help='Use extended prompt with cryptic solving guidance')
    
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
        delay=args.delay,
        efficient=args.efficient,
        batch_size=args.batch_size,
        extended_prompt=args.extended_prompt
    )
    
    # Save configuration
    config_file = output_dir / 'experiment_config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Experiment configuration saved to: {config_file}")
    
    # Run models if not skipped
    if not args.skip_run:
        print(f"\nRunning {len(args.models)} models in parallel...")
        
        results = run_models_parallel(config, model_args_map, args.max_workers)
        
        # Print summary
        print(f"\n{'='*60}")
        print("EXPERIMENT EXECUTION SUMMARY")
        print(f"{'='*60}")
        
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        print(f"Successful: {len(successful)}")
        for result in successful:
            print(f"   {result['model']}: {result['runtime']:.1f}s")
        
        if failed:
            print(f"Failed: {len(failed)}")
            for result in failed:
                print(f"   {result['model']}: {result.get('error', 'Unknown error')}")
    
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

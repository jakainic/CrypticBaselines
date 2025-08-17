import json
import argparse
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm
import time
import random

from models import get_model, BaseModel

@dataclass
class EnhancedRecord:
    """Enhanced record structure for evaluation"""
    clue: str
    length: str
    answer: Optional[str] = None
    candidates: List[Dict[str, Any]] = None
    picked: Optional[str] = None
    model_name: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

def parse_data_file(data_path: str) -> List[Dict[str, Any]]:
    """Parse input data file (CSV or JSON)"""
    examples = []
    
    if data_path.endswith('.csv'):
        import pandas as pd
        df = pd.read_csv(data_path)
        for _, row in df.iterrows():
            # Check for target_length first, then fall back to length/enumeration
            length_value = ''
            if 'target_length' in row and pd.notna(row.get('target_length')):
                length_value = str(row.get('target_length', ''))
            elif 'length' in row:
                length_value = str(row.get('length', ''))
            elif 'enumeration' in row:
                length_value = str(row.get('enumeration', ''))
                
            example = {
                'clue': str(row.get('clue', '')),
                'length': length_value,
                'answer': str(row.get('answer', ''))
            }
            examples.append(example)
    else:
        # Assume JSON format
        with open(data_path, 'r') as f:
            for line in f:
                try:
                    example = json.loads(line.strip())
                    # Map target_length first, then enumeration to length for JSON data
                    if 'target_length' in example and example['target_length'] is not None:
                        example['length'] = str(example['target_length'])
                    elif 'enumeration' in example and 'length' not in example:
                        example['length'] = example['enumeration']
                    examples.append(example)
                except json.JSONDecodeError:
                    continue
    
    return examples

def run_model_on_examples(
    model: BaseModel, 
    examples: List[Dict[str, Any]], 
    k: int = 5,
    delay: float = 0.0,
    max_examples: Optional[int] = None,
    max_retries: int = 3
) -> List[EnhancedRecord]:
    """Run model on examples and generate enhanced records"""
    
    if max_examples:
        examples = examples[:max_examples]
    
    records = []
    
    for i, example in enumerate(tqdm(examples, desc=f"Running {model.get_name()}")):
        try:
            # Generate candidates with retry logic
            raw_response = None
            candidate_dicts = []
            validation_errors = []
            retry_count = 0
            
            while retry_count < max_retries and not candidate_dicts:
                if retry_count > 0:
                    print(f"Retry {retry_count}/{max_retries} for example {i}")
                    time.sleep(delay * 2)  # Longer delay on retries
                
                raw_response = model.generate_candidates(
                    clue=example['clue'],
                    length=example.get('length', ''),
                    k=k
                )
                
                # Validate and parse the response
                candidate_dicts, validation_errors = validate_candidate_json(raw_response, k)
                retry_count += 1
            
            if not candidate_dicts:
                print(f"All {max_retries} attempts failed for example {i}: {validation_errors}")
                print(f"Final raw response: {raw_response[:200] if raw_response else 'None'}...")
                # Fall back to empty candidates
                candidate_dicts = []
            
            # Create enhanced record
            record = EnhancedRecord(
                clue=example['clue'],
                length=example.get('length', ''),
                answer=example.get('answer'),
                candidates=candidate_dicts,
                picked=candidate_dicts[0]['answer'] if candidate_dicts else None,
                model_name=model.get_name(),
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                metadata={
                    'example_index': i,
                    'num_candidates': len(candidate_dicts),
                    'target_length': parse_length(example.get('length', '')),
                    'validation_errors': validation_errors if not candidate_dicts else None,
                    'raw_response_preview': raw_response[:200] if not candidate_dicts else None,
                    'retry_count': retry_count - 1 if retry_count > 1 else 0,
                    'successful_on_retry': retry_count > 1
                }
            )
            
            records.append(record)
            
            # Add delay if specified (useful for API rate limiting)
            if delay > 0:
                time.sleep(delay)
                
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            import traceback
            # Create error record
            error_record = EnhancedRecord(
                clue=example['clue'],
                length=example.get('length', ''),
                answer=example.get('answer'),
                candidates=[],
                picked=None,
                model_name=model.get_name(),
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                metadata={
                    'example_index': i,
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'traceback': traceback.format_exc(),
                    'num_candidates': 0
                }
            )
            records.append(error_record)
    
    return records

def validate_candidate_json(content: str, k: int = 5) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Validate and parse JSON response from model.
    
    Returns:
        Tuple of (valid_candidates_dicts, validation_errors)
    """
    candidates = []
    errors = []
    
    try:
        # Try to find JSON in the response (more robust than simple regex)
        json_start = content.find('{')
        json_end = content.rfind('}')
        
        if json_start == -1 or json_end == -1:
            errors.append("No JSON structure found in response")
            return candidates, errors
        
        # Extract the JSON portion
        json_content = content[json_start:json_end + 1]
        
        # Try to parse the JSON
        try:
            data = json.loads(json_content)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON format: {e}")
            return candidates, errors
        
        # Validate the structure
        if not isinstance(data, dict):
            errors.append("Response is not a JSON object")
            return candidates, errors
        
        if 'candidates' not in data:
            errors.append("Missing 'candidates' field in response")
            return candidates, errors
        
        candidates_list = data['candidates']
        if not isinstance(candidates_list, list):
            errors.append("'candidates' field is not a list")
            return candidates, errors
        
        # Process each candidate
        for i, cand in enumerate(candidates_list[:k]):
            if not isinstance(cand, dict):
                errors.append(f"Candidate {i+1} is not a JSON object")
                continue
            
            # Extract and validate fields
            answer = cand.get('answer', '')
            confidence = cand.get('confidence', 0.5)
            reasoning = cand.get('reasoning', '')
            
            # Validate answer
            if not answer or not isinstance(answer, str):
                errors.append(f"Candidate {i+1} has invalid answer: {answer}")
                continue
            
            # Validate confidence
            if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                errors.append(f"Candidate {i+1} has invalid confidence: {confidence}")
                confidence = 0.5  # Default to 0.5 if invalid
            
            # Validate reasoning
            if reasoning and not isinstance(reasoning, str):
                errors.append(f"Candidate {i+1} has invalid reasoning type")
                reasoning = ""
            
            # Create candidate dict if valid
            candidates.append({
                'answer': answer.upper(),
                'probability': float(confidence),
                'reasoning': reasoning
            })
        
        if not candidates:
            errors.append("No valid candidates found in response")
            
    except Exception as e:
        errors.append(f"Unexpected error during JSON validation: {e}")
    
    return candidates, errors

def parse_length(length: str) -> int:
    """Parse length string to get target length for cryptic crossword answers
    
    Handles formats like:
    - (9) -> 9 (single word)
    - (5,6) -> 11 (two words: 5 + 6 = 11, but answer will have a space)
    - (3,4,2) -> 9 (three words: 3 + 4 + 2 = 9, but answer will have spaces)
    """
    if not length:
        return 0
    
    # Remove parentheses and split by commas
    clean_length = length.strip('()')
    if ',' in clean_length:
        # Multi-word answer: sum the lengths
        parts = [int(part.strip()) for part in clean_length.split(',')]
        return sum(parts)
    else:
        # Single word answer
        try:
            return int(clean_length)
        except ValueError:
            return 0

def main():
    parser = argparse.ArgumentParser(description='Enhanced harness for running models on cryptic crossword data')
    parser.add_argument('--data', required=True, help='Path to input data file (CSV or JSON)')
    parser.add_argument('--out', required=True, help='Path to output predictions file')
    parser.add_argument('--model', default='stub', choices=['stub', 'openai', 'anthropic', 'gemini'], 
                       help='Model type to use')
    parser.add_argument('--model-args', nargs='*', help='Additional model arguments (key=value format)')
    parser.add_argument('--k', type=int, default=5, help='Number of candidates to generate')
    parser.add_argument('--delay', type=float, default=0.0, help='Delay between API calls (seconds)')
    parser.add_argument('--max-examples', type=int, help='Maximum number of examples to process')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed if specified
    if args.seed is not None:
        random.seed(args.seed)
        import numpy as np
        np.random.seed(args.seed)
    
    # Parse model arguments
    model_kwargs = {}
    if args.model_args:
        for arg in args.model_args:
            if '=' in arg:
                key, value = arg.split('=', 1)
                model_kwargs[key] = value
    
    # Initialize model
    try:
        model = get_model(args.model, **model_kwargs)
        print(f"Initialized {args.model} model: {model.get_name()}")
    except Exception as e:
        print(f"Error initializing model: {e}")
        return
    
    # Load data
    print(f"Loading data from {args.data}")
    examples = parse_data_file(args.data)
    print(f"Loaded {len(examples)} examples")
    
    # Run model
    print(f"Running model with k={args.k} candidates")
    records = run_model_on_examples(
        model=model,
        examples=examples,
        k=args.k,
        delay=args.delay,
        max_examples=args.max_examples
    )
    
    # Save results
    print(f"Saving {len(records)} records to {args.out}")
    with open(args.out, 'w') as f:
        for record in records:
            f.write(json.dumps(asdict(record)) + '\n')
    
    # Print summary
    successful_records = [r for r in records if r.metadata and 'error' not in r.metadata]
    error_records = [r for r in records if r.metadata and 'error' in r.metadata]
    
    print(f"\n=== SUMMARY ===")
    print(f"Total examples processed: {len(records)}")
    print(f"Successful: {len(successful_records)}")
    print(f"Errors: {len(error_records)}")
    
    if successful_records:
        avg_candidates = sum(len(r.candidates) for r in successful_records) / len(successful_records)
        print(f"Average candidates per example: {avg_candidates:.2f}")
    
    print(f"\nResults saved to: {args.out}")

if __name__ == "__main__":
    main()

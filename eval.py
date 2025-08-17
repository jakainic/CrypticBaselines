import json
import argparse
import re
from typing import Dict, List, Any
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    total_examples: int
    top_answer_accuracy: float
    top_answer_length_accuracy: float
    candidate_coverage: float  # % of examples where any candidate was correct
    length_coverage: float     # % of examples where any candidate had correct length
    avg_candidates_per_example: float
    avg_top_probability: float
    ranking_insights: Dict[str, Any]  # insights about ranking behavior

def letters_only(s: str) -> str:
    """Extract only letters from string and convert to uppercase"""
    return re.sub(r'[^A-Za-z]', '', s or '').upper()

def parse_length(length: str) -> int:
    """Parse length string to get target length"""
    digits = re.findall(r'\d+', length or '')
    return sum(map(int, digits)) if digits else 0

def evaluate_candidates(record: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a single record with multiple candidates"""
    true_answer = record.get('answer', '')
    true_length = record.get('length', '')
    candidates = record.get('candidates', [])
    
    if not true_answer or not candidates:
        return {
            'has_true_answer': False,
            'top_correct': False,
            'top_length_correct': False,
            'any_candidate_correct': False,
            'any_candidate_length_correct': False,
            'top_probability': 0.0,
            'candidate_count': 0
        }
    
    # Pre-compute values once
    target_length = parse_length(true_length)
    true_letters = letters_only(true_answer)
    candidate_count = len(candidates)
    
    # Get top candidate info
    top_candidate = candidates[0]
    top_answer = letters_only(top_candidate.get('answer', ''))
    top_probability = top_candidate.get('probability', 0.0)

    # Compute top candidate metrics
    top_correct = top_answer == true_letters
    top_length_correct = len(top_answer) == target_length if target_length > 0 else True
    
    # Single pass through candidates with early exit
    any_candidate_correct = top_correct
    any_candidate_length_correct = top_length_correct
    
    for candidate in candidates:
        # Early exit if we found both
        if any_candidate_correct and any_candidate_length_correct:
            break
        
        candidate_answer = letters_only(candidate.get('answer', ''))
        candidate_length = len(candidate_answer)
        
        # Check correctness (can exit early if both found)
        if not any_candidate_correct and candidate_answer == true_letters:
            any_candidate_correct = True
        
        # Check length (can exit early if both found)
        if not any_candidate_length_correct and target_length > 0 and candidate_length == target_length:
            any_candidate_length_correct = True
        
    
    return {
        'has_true_answer': True,
        'top_correct': top_correct,
        'top_length_correct': top_length_correct,
        'any_candidate_correct': any_candidate_correct,
        'any_candidate_length_correct': any_candidate_length_correct,
        'top_probability': top_probability,
        'candidate_count': candidate_count,
        'target_length': target_length,
        'actual_length': len(top_answer)
    }

def calculate_ranking_insights(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate insights about model ranking behavior"""
    insights = {
        'correct_answer_ranking': {
            'top_ranked': 0,      # Correct answer was #1
            'not_top_ranked': 0,  # Correct answer was in candidates but not #1
            'total_correct': 0     # Total examples with correct answers
        }
    }
    
    for eval_result in evaluations:
        if not eval_result['has_true_answer']:
            continue
            
        if eval_result['any_candidate_correct']:
            # There was a correct answer - was it top-ranked?
            insights['correct_answer_ranking']['total_correct'] += 1
            if eval_result['top_correct']:
                insights['correct_answer_ranking']['top_ranked'] += 1
            else:
                insights['correct_answer_ranking']['not_top_ranked'] += 1
    
    return insights

def main():
    parser = argparse.ArgumentParser(description='Enhanced evaluation of cryptic crossword models')
    parser.add_argument('--pred', required=True, help='Path to predictions file')
    parser.add_argument('--report', required=True, help='Path to output report file')
    parser.add_argument('--detailed', action='store_true', help='Output detailed per-example results')
    parser.add_argument('--detailed-out', help='Path for detailed output file')
    
    args = parser.parse_args()
    
    # Read and evaluate predictions
    evaluations = []
    valid_count = 0
    invalid_count = 0
    
    with open(args.pred, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())
                eval_result = evaluate_candidates(record)
                
                if eval_result['has_true_answer']:
                    valid_count += 1
                    # Only store detailed info if requested
                    if args.detailed:
                        eval_result['line_number'] = line_num
                        eval_result['clue'] = record.get('clue', '')
                    evaluations.append(eval_result)
                else:
                    invalid_count += 1
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}: {e}")
                invalid_count += 1
                continue
    
    # Calculate aggregate metrics
    if not evaluations:
        print("No valid examples found for evaluation")
        return
    
    print(f"Processed {valid_count + invalid_count} records: {valid_count} valid, {invalid_count} invalid")
    
    metrics = EvaluationMetrics(
        total_examples=len(evaluations),
        top_answer_accuracy=np.mean([e['top_correct'] for e in evaluations]),
        top_answer_length_accuracy=np.mean([e['top_length_correct'] for e in evaluations]),
        candidate_coverage=np.mean([e['any_candidate_correct'] for e in evaluations]),
        length_coverage=np.mean([e['any_candidate_length_correct'] for e in evaluations]),
        avg_candidates_per_example=np.mean([e['candidate_count'] for e in evaluations]),
        avg_top_probability=np.mean([e['top_probability'] for e in evaluations]),
        ranking_insights=calculate_ranking_insights(evaluations)
    )
    
    # Print summary
    print("=== ENHANCED EVALUATION RESULTS ===")
    print(f"Total examples: {metrics.total_examples}")
    print(f"\n=== TOP ANSWER PERFORMANCE ===")
    print(f"Answer accuracy: {metrics.top_answer_accuracy:.3f} ({metrics.top_answer_accuracy*100:.1f}%)")
    print(f"Length accuracy: {metrics.top_answer_length_accuracy:.3f} ({metrics.top_answer_length_accuracy*100:.1f}%)")
    
    print(f"\n=== CANDIDATE COVERAGE ===")
    print(f"Any candidate correct: {metrics.candidate_coverage:.3f} ({metrics.candidate_coverage*100:.1f}%)")
    print(f"Length coverage: {metrics.length_coverage:.3f} ({metrics.length_coverage*100:.1f}%)")
    
    print(f"\n=== MODEL BEHAVIOR ===")
    print(f"Average candidates per example: {metrics.avg_candidates_per_example:.2f}")
    print(f"Average top probability: {metrics.avg_top_probability:.3f}")
    
    print(f"\n=== RANKING INSIGHTS ===")
    correct_ranking = metrics.ranking_insights['correct_answer_ranking']
    
    if correct_ranking['total_correct'] > 0:
        top_ranked_pct = correct_ranking['top_ranked'] / correct_ranking['total_correct']
        print(f"Correct answers: {correct_ranking['top_ranked']}/{correct_ranking['total_correct']} ({top_ranked_pct:.1%}) were top-ranked")
    else:
        print("No correct answers found in dataset")
    
    # Save detailed report
    report_data = {
        'summary': {
            'total_examples': metrics.total_examples,
            'top_answer_accuracy': metrics.top_answer_accuracy,
            'top_answer_length_accuracy': metrics.top_answer_length_accuracy,
            'candidate_coverage': metrics.candidate_coverage,
            'length_coverage': metrics.length_coverage,
            'avg_candidates_per_example': metrics.avg_candidates_per_example,
            'avg_top_probability': metrics.avg_top_probability,
            'ranking_insights': metrics.ranking_insights
        },
        'detailed_results': evaluations if args.detailed else None
    }
    
    with open(args.report, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    # Save detailed results if requested
    if args.detailed and args.detailed_out:
        with open(args.detailed_out, 'w') as f:
            for eval_result in evaluations:
                f.write(json.dumps(eval_result) + '\n')
    
    print(f"\nReport saved to: {args.report}")
    if args.detailed and args.detailed_out:
        print(f"Detailed results saved to: {args.detailed_out}")

if __name__ == "__main__":
    main()

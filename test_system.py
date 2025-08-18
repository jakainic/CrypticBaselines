#!/usr/bin/env python3
"""
Simple test script to verify the streamlined baseline system works correctly.
"""

import json
import tempfile
import os
from pathlib import Path

def test_models():
    """Test that models can be imported and instantiated"""
    print("Testing model imports...")
    
    try:
        from models import get_model
        print("Models imported successfully")
        
        # Test stub model
        model = get_model('stub')
        print(f"Stub model created: {model.get_name()}")
        
        # Test candidate generation
        candidates = model.generate_candidates("Test clue", k=3)
        print(f"Generated candidates response")
        print(f"Response length: {len(candidates)} characters")
        print(f"Response preview: {candidates[:100]}...")
        
        # Test other model types (if API keys available)
        try:
            if os.environ.get('OPENAI_API_KEY'):
                gpt_model = get_model('openai', model_name='gpt-4o-mini')  # Use cheaper model for testing
                print(f"OpenAI model created: {gpt_model.get_name()}")
            
            if os.environ.get('ANTHROPIC_API_KEY'):
                claude_model = get_model('anthropic', model_name='claude-3-haiku-20240307')  # Use cheaper model for testing
                print(f"Anthropic model created: {claude_model.get_name()}")
                
            # Skip Google Gemini test if package not available
            if os.environ.get('GOOGLE_API_KEY'):
                try:
                    gemini_model = get_model('gemini', model_name='gemini-1.5-flash')
                    print(f"Google Gemini model created: {gemini_model.get_name()}")
                except Exception as e:
                    print(f"Google Gemini not available: {e}")
                
        except Exception as e:
            print(f"Note: Some models require API keys: {e}")
            
    except Exception as e:
        print(f"Model test failed: {e}")
        return False
    
    return True

def test_data_parsing():
    """Test data parsing functionality"""
    print("\nTesting data parsing...")
    
    try:
        from harness import parse_data_file
        
        # Create test CSV data
        test_csv = """clue,enumeration,answer
"Test clue 1","3","CAT"
"Test clue 2","4","DOGS"
"Test clue 3","5","HORSE"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(test_csv)
            temp_file = f.name
        
        # Test parsing
        examples = parse_data_file(temp_file)
        print(f"Parsed {len(examples)} examples from CSV")
        
        # Cleanup
        os.unlink(temp_file)
        
    except Exception as e:
        print(f"Data parsing test failed: {e}")
        return False
    
    return True

def test_evaluation():
    """Test evaluation functionality"""
    print("\nTesting evaluation...")
    
    try:
        from eval import evaluate_candidates
        
        # Create test record
        test_record = {
            'clue': 'Test clue',
            'length': '3',
            'answer': 'CAT',  # This is the true answer
            'candidates': [
                {'answer': 'CAT', 'probability': 0.9},
                {'answer': 'DOG', 'probability': 0.7},
                {'answer': 'BAT', 'probability': 0.5}
            ]
        }
        
        # Test evaluation
        result = evaluate_candidates(test_record)
        print(f"Evaluation completed")
        print(f"Top correct: {result['top_correct']}")
        print(f"Any candidate correct: {result['any_candidate_correct']}")
        
    except Exception as e:
        print(f"Evaluation test failed: {e}")
        return False
    
    return True

def test_end_to_end():
    """Test end-to-end functionality with stub model"""
    print("\nTesting end-to-end functionality...")
    
    try:
        # Create test data
        test_data = [
            {"clue": "Test clue 1", "length": "3", "answer": "CAT"},
            {"clue": "Test clue 2", "length": "4", "answer": "DOGS"}
        ]
        
        # Save test data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for example in test_data:
                f.write(json.dumps(example) + '\n')
            temp_data = f.name
        
        # Test harness
        from harness import run_model_on_examples
        from models import get_model
        
        model = get_model('stub')
        records = run_model_on_examples(model, test_data, k=3, max_examples=2)
        
        print(f"Generated {len(records)} records")
        print(f"Average candidates: {sum(len(r.candidates) for r in records) / len(records):.1f}")
        
        # Test evaluation
        from eval import evaluate_candidates
        
        evaluations = []
        for record in records:
            # Convert EnhancedRecord to dict format expected by evaluate_candidates
            record_dict = {
                'clue': record.clue,
                'length': record.length,
                'answer': record.answer,
                'candidates': record.candidates
            }
            eval_result = evaluate_candidates(record_dict)
            evaluations.append(eval_result)
        
        print(f"Evaluated {len(evaluations)} records")
        
        # Cleanup
        os.unlink(temp_data)
        
    except Exception as e:
        print(f"End-to-end test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("=== TESTING STREAMLINED BASELINE SYSTEM ===\n")
    
    tests = [
        ("Model imports", test_models),
        ("Data parsing", test_data_parsing),
        ("Evaluation", test_evaluation),
        ("End-to-end", test_end_to_end)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running: {test_name}")
        if test_func():
            passed += 1
            print(f"{test_name} passed\n")
        else:
            print(f"{test_name} failed\n")
    
    print(f"=== TEST RESULTS ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("All tests passed! The system is working correctly.")
        print("\nYou can now run:")
        print("python example_usage.py")
        print("python harness.py --help")
        print("python eval.py --help")
    else:
        print("Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main()

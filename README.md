# Cryptic Crossword Baselines

Evaluation framework for comparing different language models on cryptic crossword solving.

## Purpose

Compare performance of various LLM providers against each other and baseline methods on cryptic crossword clues.

## Files

- **`harness.py`** - Core evaluation engine
- **`models.py`** - Model wrappers for different providers
- **`run_baselines.py`** - Main evaluation runner
- **`run_baselines_parallel.py`** - Parallel evaluation
- **`eval.py`** - Evaluation metrics and analysis
- **`test_system.py`** - System validation tests

## Output

- **Predictions**: Model outputs with confidence scores
- **Reports**: Accuracy metrics and analysis
- **Comparisons**: Side-by-side model performance
- **Detailed logs**: Full evaluation traces

## Future work
- Enhanced prompt engineering
- Calibration analysis

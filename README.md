# Cryptic Crossword Baselines

Evaluation framework for comparing different language models on cryptic crossword solving.

## Purpose

Compare performance of various LLM providers against each other and baseline methods on cryptic crossword clues.

## Features

- **Parallel Processing**: Run multiple models simultaneously
- **Batch Processing**: If only predicting answers
- **Confidence Probabilities**: For calibration analysis
- **Flexible Prompts**: Basic and extended cryptic solving guidance
- **Multiple Models**: Support for OpenAI, Anthropic, and Google Gemini

## Output

- **Predictions**: Model predictions (optional with confidence scores)
- **Reports**: Accuracy metrics and analysis
- **Comparisons**: Side-by-side model performance
- **Detailed logs**: Full evaluation traces

## Future work
- Enhanced prompt engineering
- Calibration analysis
- Troubleshooting: API rate limits, empty JSON

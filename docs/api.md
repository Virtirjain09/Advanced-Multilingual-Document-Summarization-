# Documentation of the main classes and functions

# API Documentation

## Main Classes

### BartSummarizer

- `__init__()`: Initializes the BART model.
- `generate_summary(text: str, length: str) -> str`: Generates a summary using BART.

### PegasusSummarizer

- `__init__()`: Initializes the PEGASUS model.
- `generate_summary(text: str, length: str) -> str`: Generates a summary using PEGASUS.

### Translator

- `translate(text: str, from_lang: str, to_lang: str) -> str`: Translates text between languages.

### ModelEvaluator

- `calculate_metrics(reference: str, summary: str) -> Dict[str, float]`: Calculates evaluation metrics for summaries.

## Utility Functions

- `extract_text_from_pdf(pdf_file) -> str`: Extracts text from a PDF file.
- `detect_language(text: str) -> Tuple[str, str]`: Detects the language of the input text.
- `plot_metrics_comparison(metrics_bart, metrics_pegasus) -> go.Figure`: Creates a comparison plot of metrics.

For detailed implementation, refer to the source code in `sumall1.py`.

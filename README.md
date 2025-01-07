# Advanced Multilingual Document Summarization

This project is a multilingual text translation and summarization tool that uses BART and PEGASUS models to summarize English input from various languages(like Hindi, Spanish, German, Turkish and French). It provides translation, summarization, and evaluation capabilities for text input and PDF files.

## Key Features

- Supports both text and PDF input formats
- Automatic language detection and translation to English
- Abstractive summarization using BART and PEGASUS models
- Customizable summary lengths (short, medium, long)
- Comparative evaluation of model performance using ROUGE and BLEU metrics
- Interactive web application for easy use and result visualization
- Downloadable results in Excel and CSV formats

## Technical Highlights

- Preprocessing pipeline for cleaning and standardizing input text
- Chunking strategies for handling long documents
- Fine-tuning of BART and PEGASUS models for domain-specific summarization
- Robust error handling and retry mechanisms for translation
- Visualization of performance metrics using Plotly
- Integration with Streamlit for a user-friendly interface

## Methodology

1. Input Preprocessing and Language Standardization
   - Text extraction from PDFs using PyPDF2
   - Language detection with langdetect
   - Translation to English using Google Translator API

2. Abstractive Summarization
   - BART model with additional heuristics for content augmentation
   - PEGASUS model with focus on beam search and length penalty adjustments
   - Segmentation techniques for long inputs

3. Evaluation and Visualization
   - Calculation of ROUGE-1, ROUGE-2, ROUGE-L, BLEU, and length ratio metrics
   - Comparative visualization of model performance

## Results

- BART showed better performance in ROUGE metrics, indicating superior content preservation
- PEGASUS demonstrated a slight advantage in BLEU scores, suggesting more fluent outputs

## Applications

- Cross-lingual information access in academic, professional, and public domains
- Domain-specific summarization for fields like healthcare, finance, and research
- Educational tool for NLP practitioners and researchers

This project bridges the gap between multilingual summarization research and practical application, offering a comprehensive solution for generating high-quality, domain-specific English summaries from multilingual inputs.

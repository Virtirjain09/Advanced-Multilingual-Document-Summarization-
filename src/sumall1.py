import streamlit as st
import PyPDF2
import pandas as pd
from langdetect import detect
import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, 
    PegasusTokenizer, PegasusForConditionalGeneration,
    pipeline
)
from deep_translator import GoogleTranslator
import time
from typing import Optional, Dict
from rouge_score import rouge_scorer
import plotly.graph_objects as go
import io
import nltk
import ssl
from nltk.translate.bleu_score import sentence_bleu

def setup_nltk():
    try:
        # Create unverified HTTPS context for NLTK downloads
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # Download required NLTK data
        nltk.download('punkt')
        return True
    except Exception as e:
        st.error(f"Failed to setup NLTK: {str(e)}")
        return False

class ModelEvaluator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def calculate_metrics(self, reference: str, summary: str) -> Dict[str, float]:
        try:
            # Preprocess texts for better ROUGE scores
            reference = self.preprocess_text(reference)
            summary = self.preprocess_text(summary)
            
            # Calculate ROUGE scores with improved preprocessing
            scores = self.scorer.score(reference, summary)
            
            # Calculate BLEU score
            reference_tokens = self.tokenize_safely(reference)
            summary_tokens = self.tokenize_safely(summary)
            
            try:
                bleu_score = sentence_bleu([reference_tokens], summary_tokens)
            except Exception:
                bleu_score = 0.0
            
            return {
                'rouge1_f1': scores['rouge1'].fmeasure,
                'rouge2_f1': scores['rouge2'].fmeasure,
                'rougeL_f1': scores['rougeL'].fmeasure,
                'bleu': bleu_score,
                'length_ratio': len(summary.split()) / len(reference.split())
            }
        except Exception as e:
            st.error(f"Metrics calculation error: {str(e)}")
            return {
                'rouge1_f1': 0.0,
                'rouge2_f1': 0.0,
                'rougeL_f1': 0.0,
                'bleu': 0.0,
                'length_ratio': 0.0
            }
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for better ROUGE scores"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Convert to lowercase for better matching
        text = text.lower()
        # Add periods if missing
        if not text.endswith('.'):
            text += '.'
        return text
    
    def tokenize_safely(self, text: str) -> list:
        try:
            return nltk.word_tokenize(text.lower())
        except Exception:
            return text.lower().split()

def clean_text(text: str) -> str:
    """Clean text of illegal characters and normalize whitespace"""
    if not text:
        return ""
    # Remove illegal characters and normalize whitespace
    text = ''.join(char for char in text if ord(char) < 128)  # Remove non-ASCII
    text = ' '.join(text.split())  # Normalize whitespace
    text = text.replace('\x00', '')  # Remove null bytes
    return text

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                # Clean the extracted text
                page_text = clean_text(page_text)
                text += page_text + " "
        return text.strip()
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

class BartSummarizer:
    def __init__(self):
        try:
            model_name = "facebook/bart-large-cnn"
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.summarizer = pipeline(
                "summarization",
                model=model_name,
                device=0 if self.device == "cuda" else -1,
                framework="pt"
            )
            self.key_phrases = set()
        except Exception as e:
            st.error(f"Error initializing BART: {str(e)}")
            raise e

    def generate_summary(self, text: str, length: str = 'medium') -> str:
        # Clean and preprocess
        text = clean_text(text)
        self.key_phrases = self.extract_key_phrases(text)
        
        # Optimized length parameters for better content retention
        text_length = len(text.split())
        length_params = {
            'short': (max(text_length // 3, 100), max(text_length // 4, 50)),
            'medium': (max(text_length // 2, 150), max(text_length // 3, 75)),
            'long': (max(text_length // 1.5, 250), max(text_length // 2, 125))
        }
        max_length, min_length = length_params[length]
        
        try:
            # Extract key sentences first
            important_content = self.extract_important_content(text)
            
            if len(important_content.split()) > 1024:
                summary = self.process_long_text(important_content, max_length, min_length)
            else:
                summary = self.summarizer(
                    important_content,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=True,
                    num_beams=8,
                    top_k=50,
                    top_p=0.95,
                    length_penalty=2.0,
                    repetition_penalty=3.0,
                )[0]['summary_text']
            
            # Enhance summary with key information
            enhanced_summary = self.enhance_summary(summary, text)
            return self.post_process_summary(enhanced_summary)
            
        except Exception as e:
            st.error(f"BART Summarization error: {str(e)}")
            return text[:1000]

    def extract_key_phrases(self, text: str) -> set:
        """Extract important phrases and entities"""
        words = text.lower().split()
        phrases = set()
        
        # Single words (frequency-based)
        word_freq = {}
        for word in words:
            if len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Add top frequent words
        phrases.update(word for word, freq in 
                      sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20])
        
        # Add noun phrases (simple heuristic)
        sentences = text.split('. ')
        for sentence in sentences:
            words = sentence.split()
            for i in range(len(words) - 1):
                if words[i][0].isupper() and words[i+1][0].isupper():
                    phrases.add(f"{words[i]} {words[i+1]}".lower())
        
        return phrases

    def extract_important_content(self, text: str) -> str:
        sentences = text.split('. ')
        scored_sentences = []
        
        for idx, sentence in enumerate(sentences):
            score = 0
            # Position score
            if idx < len(sentences) * 0.3:  # First 30%
                score += 2
            elif idx < len(sentences) * 0.8:  # First 80%
                score += 1
            
            # Content score
            score += sum(3 for phrase in self.key_phrases 
                        if phrase in sentence.lower())
            
            # Length score
            words = len(sentence.split())
            if 10 <= words <= 30:
                score += 1
            
            scored_sentences.append((sentence, score))
        
        # Select top 70% of sentences
        threshold = len(sentences) * 0.7
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        selected = [sent for sent, _ in scored_sentences[:int(threshold)]]
        
        return '. '.join(selected) + '.'

    def enhance_summary(self, summary: str, original: str) -> str:
        summary_sentences = summary.split('. ')
        enhanced = summary_sentences.copy()
        
        # Add missing key information
        for phrase in self.key_phrases:
            if phrase not in summary.lower():
                original_sentences = original.split('. ')
                for sent in original_sentences:
                    if phrase in sent.lower() and sent not in enhanced:
                        enhanced.append(sent)
                        break
        
        return '. '.join(enhanced) + '.'

    def post_process_summary(self, summary: str) -> str:
        sentences = summary.split('. ')
        processed = []
        
        for sent in sentences:
            if sent:
                # Fix capitalization
                sent = sent[0].upper() + sent[1:] if len(sent) > 1 else sent
                # Remove redundant spaces
                sent = ' '.join(sent.split())
                processed.append(sent)
        
        return '. '.join(processed) + '.'
    
    def process_long_text(self, text: str, max_length: int, min_length: int) -> str:
        sentences = text.split('. ')
        chunks = ['. '.join(sentences[i:i + 20]) for i in range(0, len(sentences), 20)]
    
    # Process each chunk
        chunk_summaries = []
        for chunk in chunks:
            if len(chunk.split()) > 50:  # Only process chunks with sufficient content
                summary = self.summarizer(
                    chunk,
                    max_length=max_length // len(chunks),
                    min_length=min_length // len(chunks),
                    do_sample=False,
                    num_beams=4,
                    length_penalty=2.0,
                )[0]['summary_text']
                chunk_summaries.append(summary)
    
    # Combine and summarize again
        combined_summary = ' '.join(chunk_summaries)
        final_summary = self.summarizer(
            combined_summary,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            num_beams=4,
            length_penalty=2.0,
        )[0]['summary_text']
    
        return final_summary.strip()
    
    

class PegasusSummarizer:
    
    def __init__(self):
        try:
            # Use a different pre-trained model that's fully initialized
            model_name = "google/pegasus-large"  # or "google/pegasus-cnn_dailymail"
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load with position embeddings
            self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
            self.model = PegasusForConditionalGeneration.from_pretrained(
                model_name,
                load_in_8bit=False,  # Disable 8-bit loading
                device_map="auto" if torch.cuda.is_available() else None
            ).to(self.device)
            
            # Ensure model is in evaluation mode
            self.model.eval()
            
        except Exception as e:
            st.error(f"Error initializing PEGASUS: {str(e)}")
            raise e

    def generate_summary(self, text: str, length: str = 'medium') -> str:
        try:
            text = clean_text(text)
            
            # Extract key information first
            key_info = self.extract_key_information(text)
            
            # Adjusted length parameters
            text_length = len(text.split())
            length_params = {
                'short': (max(text_length // 4, 100), max(text_length // 5, 50)),
                'medium': (max(text_length // 3, 150), max(text_length // 4, 75)),
                'long': (max(text_length // 2, 250), max(text_length // 3, 125))
            }
            max_length, min_length = length_params[length]
            
            if len(key_info.split()) > 512:
                summary = self.process_long_text(key_info, max_length, min_length)
            else:
                inputs = self.tokenizer(
                    key_info,
                    max_length=1024,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=8,
                    length_penalty=2.0,
                    repetition_penalty=3.0,
                    early_stopping=True
                )
                
                summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            # Enhance and clean the summary
            enhanced_summary = self.enhance_summary(summary, text)
            return self.post_process_summary(enhanced_summary)
            
        except Exception as e:
            st.error(f"PEGASUS Summarization error: {str(e)}")
            return text[:500]

    def extract_key_information(self, text: str) -> str:
        sentences = text.split('. ')
        scored_sentences = []
        
        # Create a simple TF-IDF-like scoring
        word_freq = {}
        total_sentences = len(sentences)
        
        for sentence in sentences:
            words = set(word.lower() for word in sentence.split() if len(word) > 3)
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        for idx, sentence in enumerate(sentences):
            score = 0
            words = set(word.lower() for word in sentence.split() if len(word) > 3)
            
            # TF-IDF-like score
            score += sum(1/word_freq[word] for word in words)
            
            # Position score
            if idx < len(sentences) * 0.2:  # First 20%
                score *= 1.5
            
            scored_sentences.append((sentence, score))
        
        # Select top 60% of sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        selected = [sent for sent, _ in scored_sentences[:int(total_sentences * 0.6)]]
        
        return '. '.join(selected) + '.'

    def enhance_summary(self, summary: str, original: str) -> str:
        summary_words = set(summary.lower().split())
        important_sentences = []
        
        # Find sentences with important information not in summary
        for sentence in original.split('. '):
            sentence_words = set(sentence.lower().split())
            important_words = sentence_words - summary_words
            
            if len(important_words) > 3:  # If sentence has significant new information
                importance_score = sum(1 for word in important_words if len(word) > 3)
                if importance_score > 2:
                    important_sentences.append(sentence)
        
        # Add most important missing sentences
        if important_sentences:
            summary += ' ' + '. '.join(important_sentences[:2]) + '.'
        
        return summary

    def post_process_summary(self, summary: str) -> str:
        sentences = summary.split('. ')
        processed = []
        seen_content = set()
        
        for sent in sentences:
            # Remove duplicates and short sentences
            content = ' '.join(sent.lower().split())
            if content not in seen_content and len(sent.split()) > 5:
                processed.append(sent[0].upper() + sent[1:] if len(sent) > 1 else sent)
                seen_content.add(content)
        
        return '. '.join(processed) + '.'
    
    def process_long_text(self, text: str, max_length: int, min_length: int) -> str:

        try:
        # Split into smaller chunks safely
            tokens = self.tokenizer.encode(text)
            chunk_size = 512  # Safe chunk size
            chunks = []
        
            for i in range(0, len(tokens), chunk_size):
                chunk_tokens = tokens[i:i + chunk_size]
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                if len(chunk_text.split()) > 50:  # Only process meaningful chunks
                    chunks.append(chunk_text)
        
        # Process each chunk
            chunk_summaries = []
            for chunk in chunks:
                inputs = self.tokenizer(
                    chunk,
                    max_length=1024,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
            
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_length // len(chunks),
                    min_length=min_length // len(chunks),
                    num_beams=4,
                    length_penalty=1.0,
                    early_stopping=True
                )
            
                chunk_summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                if chunk_summary:
                    chunk_summaries.append(chunk_summary)
        
            if not chunk_summaries:
                return text[:max_length]
            
        # Combine and summarize again
            combined_summary = ' '.join(chunk_summaries)
            inputs = self.tokenizer(
                combined_summary,
                max_length=1024,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
        
            final_summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True
            )
        
            return self.tokenizer.decode(final_summary_ids[0], skip_special_tokens=True).strip()
            
        except Exception as e:
            st.error(f"Long text processing error: {str(e)}")
            return text[:max_length]


class Translator:
    def __init__(self):
        self.max_retries = 3
        self.delay_between_retries = 2  # seconds
    
    def translate(self, text: str, from_lang: str, to_lang: str = 'en') -> str:
        if from_lang == to_lang:
            return text
            
        for attempt in range(self.max_retries):
            try:
                translator = GoogleTranslator(source=from_lang, target=to_lang)
                
                # Split text into smaller chunks to avoid connection issues
                chunk_size = 1000  # Reduced chunk size
                chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
                translated_chunks = []
                
                for chunk in chunks:
                    try:
                        translated_chunk = translator.translate(chunk)
                        if translated_chunk:
                            translated_chunks.append(translated_chunk)
                        time.sleep(1)  # Increased delay between chunks
                    except Exception as e:
                        st.warning(f"Warning: Chunk translation failed, retrying... ({str(e)})")
                        time.sleep(self.delay_between_retries)
                        continue
                
                if translated_chunks:
                    return ' '.join(translated_chunks)
                    
            except Exception as e:
                if attempt < self.max_retries - 1:
                    st.warning(f"Translation attempt {attempt + 1} failed, retrying... ({str(e)})")
                    time.sleep(self.delay_between_retries)
                else:
                    st.error("Translation failed after multiple attempts. Please try again later.")
                    return text  # Return original text if translation fails
        
        return text  # Return original text if all attempts fail

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def detect_language(text: str):
    try:
        lang_code = detect(text)
        language_names = {
            'en': 'English', 'es': 'Spanish', 'fr': 'French',
            'de': 'German', 'it': 'Italian', 'pt': 'Portuguese',
            'nl': 'Dutch', 'ru': 'Russian', 'ar': 'Arabic',
            'hi': 'Hindi', 'zh-cn': 'Chinese', 'ja': 'Japanese'
        }
        return lang_code, language_names.get(lang_code, lang_code)
    except Exception as e:
        st.error(f"Error detecting language: {str(e)}")
        return None, None

def plot_metrics_comparison(metrics_bart, metrics_pegasus):
    metrics = ['rouge1_f1', 'rouge2_f1', 'rougeL_f1', 'bleu']
    
    fig = go.Figure(data=[
        go.Bar(name='BART', x=metrics, y=[metrics_bart[m] for m in metrics]),
        go.Bar(name='PEGASUS', x=metrics, y=[metrics_pegasus[m] for m in metrics])
    ])
    
    fig.update_layout(
        title='Model Comparison Metrics',
        xaxis_title='Metric',
        yaxis_title='Score',
        barmode='group',
        template='plotly_white'
    )
    
    return fig

def main():
    st.set_page_config(page_title="Advanced Document Processor", layout="wide")
    
    # Setup NLTK at startup
    setup_nltk()
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {max-width: 1200px; margin: 0 auto;}
        .stTabs [data-baseweb="tab-list"] {gap: 24px;}
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f0f2f6;
            border-radius: 4px;
            padding: 10px 20px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1f77b4;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🔤 Advanced Document Processor")
    st.markdown("### Multilingual Text Processing with BART & PEGASUS")
    
    # Initialize components
    if 'translator' not in st.session_state:
        st.session_state.translator = Translator()
    if 'bart_summarizer' not in st.session_state:
        st.session_state.bart_summarizer = BartSummarizer()
    if 'pegasus_summarizer' not in st.session_state:
        st.session_state.pegasus_summarizer = PegasusSummarizer()
    if 'evaluator' not in st.session_state:
        st.session_state.evaluator = ModelEvaluator()
    
    # Input section
    st.subheader("📄 Document Input")
    input_type = st.radio("Select input type:", ("Text Input", "PDF File"), horizontal=True)
    
    input_text = ""
    if input_type == "Text Input":
        input_text = st.text_area("Enter your text here:", height=200)
    else:
        uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
        if uploaded_file:
            with st.spinner("Processing PDF..."):
                input_text = extract_text_from_pdf(uploaded_file)
    
    if input_text:
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Original", "🔄 Translation", "📊 Summarization", "📈 Evaluation"])
        
        with tab1:
            st.text_area("Original Text", value=input_text, height=200)
            lang_code, lang_name = detect_language(input_text)
            if lang_code:
                st.info(f"📌 Detected Language: {lang_name} ({lang_code})")
        
        with tab2:
            if lang_code and lang_code != 'en':
                if st.button("🔄 Translate to English"):
                    with st.spinner("Translating..."):
                        translated_text = st.session_state.translator.translate(input_text, lang_code)
                        if translated_text:
                            st.session_state['translated_text'] = translated_text
            
            if 'translated_text' in st.session_state:
                st.text_area("English Translation", value=st.session_state['translated_text'], height=200)
            elif lang_code == 'en':
                st.info("✓ Text is already in English")
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("BART Summary")
                summary_type_bart = st.radio("BART summary length:", 
                                           ["short", "medium", "long"], 
                                           horizontal=True,
                                           key="bart_length")
                if st.button("Generate BART Summary"):
                    with st.spinner("Generating BART summary..."):
                        text_to_summarize = st.session_state.get('translated_text', input_text)
                        summary = st.session_state.bart_summarizer.generate_summary(
                            text_to_summarize, 
                            length=summary_type_bart
                        )
                        if summary:
                            st.session_state['bart_summary'] = summary
                
                if 'bart_summary' in st.session_state:
                    st.text_area("BART Summary", value=st.session_state['bart_summary'], height=200)
            
            with col2:
                st.subheader("PEGASUS Summary")
                summary_type_pegasus = st.radio("PEGASUS summary length:", 
                                              ["short", "medium", "long"], 
                                              horizontal=True,
                                              key="pegasus_length")
                
                if st.button("Generate PEGASUS Summary"):
                    with st.spinner("Generating PEGASUS summary..."):
                        text_to_summarize = st.session_state.get('translated_text', input_text)
                        summary = st.session_state.pegasus_summarizer.generate_summary(
                            text_to_summarize, 
                            length=summary_type_pegasus
                        )
                        if summary:
                            st.session_state['pegasus_summary'] = summary
                
                if 'pegasus_summary' in st.session_state:
                    st.text_area("PEGASUS Summary", value=st.session_state['pegasus_summary'], height=200)
        
        with tab4:
            if 'bart_summary' in st.session_state and 'pegasus_summary' in st.session_state:
                text_to_evaluate = st.session_state.get('translated_text', input_text)
                
                metrics_bart = st.session_state.evaluator.calculate_metrics(
                    text_to_evaluate, 
                    st.session_state['bart_summary']
                )
                
                metrics_pegasus = st.session_state.evaluator.calculate_metrics(
                    text_to_evaluate, 
                    st.session_state['pegasus_summary']
                )
                
                st.plotly_chart(plot_metrics_comparison(metrics_bart, metrics_pegasus))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("BART Metrics")
                    for metric, value in metrics_bart.items():
                        st.metric(metric, f"{value:.4f}")
                
                with col2:
                    st.subheader("PEGASUS Metrics")
                    for metric, value in metrics_pegasus.items():
                        st.metric(metric, f"{value:.4f}")
            else:
                st.info("Generate both summaries to see evaluation metrics")
        
        # Download section
        if 'bart_summary' in st.session_state and 'pegasus_summary' in st.session_state:
            results_df = pd.DataFrame({
                'Content Type': ['Original Text', 'Translation', 'BART Summary', 'PEGASUS Summary'],
                'Text': [
                    input_text, 
                    st.session_state.get('translated_text', ""), 
                    st.session_state['bart_summary'],
                    st.session_state['pegasus_summary']
                ]
            })
            
            # Add metrics to the DataFrame if available
            metrics_bart = st.session_state.evaluator.calculate_metrics(
                text_to_evaluate, 
                st.session_state['bart_summary']
            )
            metrics_pegasus = st.session_state.evaluator.calculate_metrics(
                text_to_evaluate, 
                st.session_state['pegasus_summary']
            )
            
            metrics_df = pd.DataFrame({
                'Metric': list(metrics_bart.keys()),
                'BART Score': list(metrics_bart.values()),
                'PEGASUS Score': list(metrics_pegasus.values())
            })
            
            # Create Excel file with multiple sheets
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name='Summaries', index=False)
                metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
            
            # Offer Excel download
            st.download_button(
                "📥 Download Complete Results (Excel)",
                output.getvalue(),
                "document_processing_results.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key='download-excel'
            )
            
            # Also offer CSV for simpler format
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Summaries (CSV)",
                csv,
                "document_processing_results.csv",
                "text/csv",
                key='download-csv'
            )

if __name__ == "__main__":
    main()
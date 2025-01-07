# Detailed installation instructions

## Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

## Steps

1. Clone the repository: git clone https://github.com/yourusername/advanced-document-processor.git
cd advanced-document-processor

2. Create a virtual environment:
python -m venv venv
source venv/bin/activate # On Windows, use venv\Scripts\activate
text

3. Install the required packages:
pip install -r requirements.txt
text

4. Download NLTK data:
python -c "import nltk; nltk.download('punkt')"
text

5. Run the Streamlit app:
streamlit run sumall1.py
text

The application should now be running on `http://localhost:8501`.

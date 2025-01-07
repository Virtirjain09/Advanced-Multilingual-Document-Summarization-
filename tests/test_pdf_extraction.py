# Test the PDF text extraction function

import unittest
import io
from sumall1 import extract_text_from_pdf
from PyPDF2 import PdfWriter

class TestPDFExtraction(unittest.TestCase):
    def setUp(self):
        # Create a simple PDF file in memory
        self.pdf_content = io.BytesIO()
        pdf_writer = PdfWriter()
        page = pdf_writer.add_blank_page(width=200, height=200)
        page.insert_text(text="Test PDF content")
        pdf_writer.write(self.pdf_content)
        self.pdf_content.seek(0)

    def test_extract_text_from_pdf(self):
        extracted_text = extract_text_from_pdf(self.pdf_content)
        self.assertIsNotNone(extracted_text)
        self.assertIn("Test PDF content", extracted_text)

if __name__ == '__main__':
    unittest.main()

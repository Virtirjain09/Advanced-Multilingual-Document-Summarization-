# Test the language detection function

import unittest
from sumall1 import detect_language

class TestLanguageDetection(unittest.TestCase):
    def test_detect_english(self):
        lang_code, lang_name = detect_language("Hello, this is a test.")
        self.assertEqual(lang_code, 'en')
        self.assertEqual(lang_name, 'English')

    def test_detect_spanish(self):
        lang_code, lang_name = detect_language("Hola, esto es una prueba.")
        self.assertEqual(lang_code, 'es')
        self.assertEqual(lang_name, 'Spanish')

    def test_detect_french(self):
        lang_code, lang_name = detect_language("Bonjour, c'est un test.")
        self.assertEqual(lang_code, 'fr')
        self.assertEqual(lang_name, 'French')

if __name__ == '__main__':
    unittest.main()

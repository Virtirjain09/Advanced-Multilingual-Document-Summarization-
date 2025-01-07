# Test the Translator class

import unittest
from sumall1 import Translator

class TestTranslator(unittest.TestCase):
    def setUp(self):
        self.translator = Translator()
        self.test_text = "Hello, world!"

    def test_translate_to_french(self):
        translated = self.translator.translate(self.test_text, from_lang='en', to_lang='fr')
        self.assertIsNotNone(translated)
        self.assertNotEqual(translated, self.test_text)

    def test_translate_same_language(self):
        translated = self.translator.translate(self.test_text, from_lang='en', to_lang='en')
        self.assertEqual(translated, self.test_text)

if __name__ == '__main__':
    unittest.main()

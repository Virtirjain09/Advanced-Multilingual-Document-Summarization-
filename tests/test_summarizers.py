# Test the BartSummarizer and PegasusSummarizer classes

import unittest
from sumall1 import BartSummarizer, PegasusSummarizer

class TestSummarizers(unittest.TestCase):
    def setUp(self):
        self.bart_summarizer = BartSummarizer()
        self.pegasus_summarizer = PegasusSummarizer()
        self.test_text = "This is a long piece of text that needs to be summarized. It contains multiple sentences and should be long enough to test the summarization capabilities of both BART and PEGASUS models."

    def test_bart_summarizer(self):
        summary = self.bart_summarizer.generate_summary(self.test_text, length='short')
        self.assertIsNotNone(summary)
        self.assertLess(len(summary), len(self.test_text))

    def test_pegasus_summarizer(self):
        summary = self.pegasus_summarizer.generate_summary(self.test_text, length='short')
        self.assertIsNotNone(summary)
        self.assertLess(len(summary), len(self.test_text))

if __name__ == '__main__':
    unittest.main()

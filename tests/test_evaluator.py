# Test the ModelEvaluator class

import unittest
from sumall1 import ModelEvaluator

class TestModelEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = ModelEvaluator()
        self.reference = "The quick brown fox jumps over the lazy dog."
        self.summary = "A fox jumps over a dog."

    def test_calculate_metrics(self):
        metrics = self.evaluator.calculate_metrics(self.reference, self.summary)
        self.assertIn('rouge1_f1', metrics)
        self.assertIn('rouge2_f1', metrics)
        self.assertIn('rougeL_f1', metrics)
        self.assertIn('bleu', metrics)
        self.assertIn('length_ratio', metrics)

    def test_preprocess_text(self):
        processed = self.evaluator.preprocess_text("  This is A TEST.  ")
        self.assertEqual(processed, "this is a test.")

if __name__ == '__main__':
    unittest.main()

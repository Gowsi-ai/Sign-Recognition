"""
Automated Unit and Integration Tests for Unified ASL Alphabet & WLASL Pipeline
"""

import os
import unittest
import numpy as np
import pickle
from inference_engine import SignBridgeInferenceEngine
from train_asl_wlasl_unified import extract_scale_invariant_features, create_base_landmarks

class TestUnifiedSignBridgeML(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = "phrase_model_v2.p"
        cls.engine = SignBridgeInferenceEngine(
            model_path=cls.model_path,
            rejection_threshold=0.45,
            stability_count=3
        )

    def test_model_loading_and_classes(self):
        """Verify model loads all 42 classes (26 ASL alphabet + 16 WLASL)."""
        self.assertEqual(len(self.engine.classes), 42)
        self.assertEqual(len(self.engine.alphabet_classes), 26)
        self.assertEqual(len(self.engine.wlasl_classes), 16)
        # Check standard ASL alphabet letters
        for letter in ["A", "B", "C", "L", "V", "Y", "Z"]:
            self.assertIn(letter, self.engine.alphabet_classes)
        # Check core WLASL words
        for word in ["Hello", "Thank you", "Please", "Help", "Good", "Bad", "Eat", "Want"]:
            self.assertIn(word, self.engine.wlasl_classes)

    def test_feature_extraction(self):
        """Verify extract_scale_invariant_features produces 101 features without NaN or Inf."""
        pts = create_base_landmarks("A")
        feats = extract_scale_invariant_features(pts)
        self.assertEqual(len(feats), 101)
        self.assertFalse(np.isnan(feats).any(), "Features contain NaN")
        self.assertFalse(np.isinf(feats).any(), "Features contain Inf")

    def test_mode_switching(self):
        """Verify engine filters predictions accurately per mode."""
        self.assertTrue(self.engine.set_mode("alphabet"))
        self.assertEqual(self.engine.mode, "alphabet")
        self.assertTrue(self.engine.set_mode("wlasl"))
        self.assertEqual(self.engine.mode, "wlasl")
        self.assertTrue(self.engine.set_mode("all"))
        self.assertEqual(self.engine.mode, "all")

    def test_empty_frame_handling(self):
        """Verify engine handles blank frames safely without crashing."""
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.engine.process_frame(blank_frame)
        self.assertFalse(result["hand_detected"])
        self.assertIsNone(result["landmarks"])
        self.assertEqual(result["framing_status"], "searching")

    def test_direct_alphabet_prediction(self):
        """Verify real held-out test samples predict their respective letters in alphabet mode."""
        import csv
        self.engine.set_mode("alphabet")
        
        # Load samples from test_dataset.csv
        test_samples = {}
        with open("dataset/test_dataset.csv", "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                lbl = row[-1]
                if lbl in ["A", "B", "L"] and lbl not in test_samples:
                    test_samples[lbl] = [float(x) for x in row[:-1]]
                if len(test_samples) == 3:
                    break

        for test_letter, feats in test_samples.items():
            probs = self.engine.model.predict_proba([feats])[0]
            valid_indices = [self.engine.class_to_idx[c] for c in self.engine.alphabet_classes]
            sub_probs = [probs[idx] for idx in valid_indices]
            predicted = self.engine.alphabet_classes[int(np.argmax(sub_probs))]
            self.assertEqual(predicted, test_letter, f"Expected {test_letter}, got {predicted}")

    def test_direct_wlasl_prediction(self):
        """Verify real held-out test samples predict their respective words in wlasl mode."""
        import csv
        self.engine.set_mode("wlasl")

        test_samples = {}
        with open("dataset/test_dataset.csv", "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                lbl = row[-1]
                if lbl in ["Hello", "Good", "Bad", "Eat"] and lbl not in test_samples:
                    test_samples[lbl] = [float(x) for x in row[:-1]]
                if len(test_samples) == 4:
                    break

        for test_word, feats in test_samples.items():
            probs = self.engine.model.predict_proba([feats])[0]
            valid_indices = [self.engine.class_to_idx[c] for c in self.engine.wlasl_classes]
            sub_probs = [probs[idx] for idx in valid_indices]
            predicted = self.engine.wlasl_classes[int(np.argmax(sub_probs))]
            self.assertEqual(predicted, test_word, f"Expected {test_word}, got {predicted}")

if __name__ == "__main__":
    unittest.main()

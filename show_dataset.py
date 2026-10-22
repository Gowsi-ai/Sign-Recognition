"""
SignBridge AI - Dataset Inspector for Mentors & Project Review
Run this script: python show_dataset.py
Displays the exact dataset distributions, train/test split, and features.
"""

import os
import csv
import json

DATASET_DIR = "dataset"
TRAIN_PATH = os.path.join(DATASET_DIR, "train_dataset.csv")
TEST_PATH = os.path.join(DATASET_DIR, "test_dataset.csv")
SUMMARY_PATH = os.path.join(DATASET_DIR, "dataset_summary.json")

def inspect_dataset():
    print("=" * 70)
    print("      SIGNBRIDGE AI - UNIFIED ASL & WLASL DATASET INSPECTOR")
    print("=" * 70)

    # 1. Check Files on Disk
    print("\n[1] DATASET FILES ON DISK:")
    for fname in ["train_dataset.csv", "test_dataset.csv", "dataset_summary.json", "asl_alphabet_raw.csv"]:
        fpath = os.path.join(DATASET_DIR, fname)
        if os.path.exists(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"  [OK] {fname:<24} Location: dataset/{fname} ({size_mb:.2f} MB)")
        else:
            print(f"  [--] {fname:<24} Missing")

    # 2. Dataset Summary & Metrics
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
            summary = json.load(f)
        print("\n[2] DATASET SPECIFICATIONS & EVALUATION:")
        print(f"  * Dataset Name:        {summary.get('dataset_name')}")
        print(f"  * Total Samples:       {summary.get('total_samples'):,} real hand landmark samples")
        print(f"  * Training Samples:    {summary.get('train_samples'):,} (80% Stratified Split)")
        print(f"  * Testing Samples:     {summary.get('test_samples'):,} (20% Held-Out Test Split)")
        print(f"  * Held-Out Accuracy:   {summary.get('test_accuracy') * 100:.2f}% on unseen test data")
        print(f"  * Classes Count:       {summary.get('classes_count')} classes")

    # 3. Class Counts in Train & Test
    print("\n[3] CLASS DISTRIBUTION ACROSS TRAIN & TEST SETS:")
    train_counts = {}
    if os.path.exists(TRAIN_PATH):
        with open(TRAIN_PATH, "r", encoding="utf-8") as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                if row:
                    lbl = row[-1]
                    train_counts[lbl] = train_counts.get(lbl, 0) + 1

    test_counts = {}
    if os.path.exists(TEST_PATH):
        with open(TEST_PATH, "r", encoding="utf-8") as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                if row:
                    lbl = row[-1]
                    test_counts[lbl] = test_counts.get(lbl, 0) + 1

    alphabet_classes = sorted([c for c in train_counts.keys() if len(c) == 1 and c.isupper()])
    wlasl_classes = sorted([c for c in train_counts.keys() if c not in alphabet_classes])

    print(f"\n  A. ASL Manual Alphabet (26 Letters A-Z):")
    for i in range(0, len(alphabet_classes), 6):
        batch = alphabet_classes[i:i+6]
        line = "    ".join([f"'{c}': {train_counts.get(c, 0)} train / {test_counts.get(c, 0)} test" for c in batch])
        print(f"    {line}")

    print(f"\n  B. Word-Level ASL (16 Words):")
    for i in range(0, len(wlasl_classes), 4):
        batch = wlasl_classes[i:i+4]
        line = "  |  ".join([f"{w:<10}: {train_counts.get(w, 0)} train / {test_counts.get(w, 0)} test" for w in batch])
        print(f"    {line}")

    # 4. Show Sample Data Row
    print("\n[4] SAMPLE ROW STRUCTURE (101 Scale-Invariant Features + Label):")
    if os.path.exists(TRAIN_PATH):
        with open(TRAIN_PATH, "r", encoding="utf-8") as f:
            r = csv.reader(f)
            header = next(r)
            row = next(r)
            print(f"  * Total Columns: {len(header)} (f_0 to f_100 + label)")
            print(f"  * First 5 Features (f_0..f_4):   {', '.join([f'{float(x):.4f}' for x in row[:5]])}")
            print(f"  * Digit Distances (f_63..f_67):  {', '.join([f'{float(x):.4f}' for x in row[63:68]])}")
            print(f"  * Palm Normal (f_98..f_100):     {', '.join([f'{float(x):.4f}' for x in row[98:101]])}")
            print(f"  * Target Class Label:            '{row[-1]}'")

    print("\n" + "=" * 70)
    print("HOW TO SHOW THIS TO YOUR MENTOR:")
    print("1. In Terminal:      Run 'python show_dataset.py' to show this live summary.")
    print("2. In VS Code:       Click on 'dataset/train_dataset.csv' in the left file tree.")
    print("3. In Windows Excel: Open folder 'dataset' and double-click 'train_dataset.csv'.")
    print("=" * 70)

if __name__ == "__main__":
    inspect_dataset()


import pickle
from collections import Counter

with open("phrases_data.pickle", "rb") as f:
    dataset = pickle.load(f)

sequences = dataset["sequences"]
labels = dataset["labels"]

print("Total recorded sequences:", len(sequences))
print("Recorded phrases:")

for phrase, count in Counter(labels).items():
    print(f"{phrase}: {count} samples")

print("\nFirst sequence shape:", len(sequences[0]), "frames")
print("Features in first frame:", len(sequences[0][0]))

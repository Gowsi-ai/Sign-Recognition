import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_FILE = "phrases_data.pickle"
MODEL_FILE = "phrase_model.p"

with open(DATA_FILE, "rb") as f:
    dataset = pickle.load(f)

sequences = np.asarray(dataset["sequences"], dtype=np.float32)
labels = np.asarray(dataset["labels"])

print("Sequences shape:", sequences.shape)
print("Phrases found:", np.unique(labels, return_counts=True))

# Flatten each 40-frame sequence into one feature vector
X = sequences.reshape(len(sequences), -1)
y = labels

if len(np.unique(y)) < 2:
    print(
        "\nOnly one phrase was recorded. "
        "Collect samples for at least one more phrase "
        "before training a classifier."
    )
    raise SystemExit

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Test accuracy:", accuracy_score(y_test, predictions))

with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)

print("Model saved to", MODEL_FILE)
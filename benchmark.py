import pickle
import numpy as np
import time
import json
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

DATA_FILE = "phrases_data.pickle"

with open(DATA_FILE, "rb") as f:
    dataset = pickle.load(f)

sequences = np.asarray(dataset["sequences"], dtype=np.float32) # shape: (50, 40, 66)
labels = np.asarray(dataset["labels"])

print(f"Loaded dataset: {sequences.shape} with {len(labels)} labels.")
unique, counts = np.unique(labels, return_counts=True)
print(f"Classes: {dict(zip(unique, counts))}")

# 1. Feature representations:
# Representation A: Baseline flattened raw features (40 * 66 = 2640)
X_base = sequences.reshape(len(sequences), -1)

# Representation B: Enhanced Normalized & Temporal Dynamics features
# For each 40-frame sequence:
#  - Hand scale invariance: compute bounding box diameter of landmarks per frame and scale coordinates.
#  - Velocity / Delta: difference between consecutive frames (t - (t-1)) to capture motion speed & direction
#  - Aggregated motion statistics (mean, std, min, max velocities) + sampled landmark trajectories
def extract_enhanced_features(seqs):
    enhanced_list = []
    for seq in seqs: # seq shape: (40, 66)
        # seq has 66 values per frame:
        # [wrist_x, wrist_y, wrist_z, (pt0_x - wrist_x), ... (pt20_z - wrist_z)]
        # Let's extract velocities across frames
        diffs = np.diff(seq, axis=0) # shape: (39, 66)
        
        # Calculate summary statistics of motion
        vel_mean = np.mean(diffs, axis=0) # 66
        vel_std = np.std(diffs, axis=0)   # 66
        vel_max = np.max(diffs, axis=0)   # 66
        
        # Normalize landmark scale by hand bounding box across frames
        # landmarks start at index 3 (21 points * 3 = 63 relative coords)
        hand_pts = seq[:, 3:] # (40, 63)
        # Compute hand size estimate per frame
        # (max_coord - min_coord) to normalize hand distance from camera
        reshaped_pts = hand_pts.reshape(len(seq), 21, 3)
        bbox_diag = np.linalg.norm(reshaped_pts.max(axis=1) - reshaped_pts.min(axis=1), axis=1, keepdims=True)
        bbox_diag = np.where(bbox_diag == 0, 1.0, bbox_diag) # prevent div by zero
        normalized_hand_pts = (reshaped_pts / bbox_diag[:, :, np.newaxis]).reshape(len(seq), 63)
        
        # Normalized wrist trajectory relative to start frame
        wrist_traj = seq[:, :3] - seq[0, :3]
        
        norm_seq = np.hstack([wrist_traj, normalized_hand_pts]) # (40, 66)
        
        # Combined feature vector:
        # Flattened normalized trajectory (40 * 66 = 2640) + motion dynamics (66 * 3 = 198)
        feat_vector = np.concatenate([norm_seq.flatten(), vel_mean, vel_std, vel_max])
        enhanced_list.append(feat_vector)
    return np.asarray(enhanced_list, dtype=np.float32)

X_enhanced = extract_enhanced_features(sequences)
print(f"X_base shape: {X_base.shape}, X_enhanced shape: {X_enhanced.shape}")

# Stratified 5-Fold Cross Validation Evaluation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Baseline_RandomForest": (RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"), X_base),
    "Enhanced_RandomForest": (RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"), X_enhanced),
    "Enhanced_ExtraTrees": (ExtraTreesClassifier(n_estimators=200, random_state=42, class_weight="balanced"), X_enhanced),
    "Enhanced_SVM": (SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced"), X_enhanced),
    "Enhanced_KNN": (KNeighborsClassifier(n_neighbors=5, weights="distance"), X_enhanced)
}

results = {}

for name, (clf, X_data) in models.items():
    t0 = time.time()
    y_pred = cross_val_predict(clf, X_data, labels, cv=skf)
    latency = (time.time() - t0) / len(X_data) * 1000 # ms per sample
    
    acc = accuracy_score(labels, y_pred)
    f1_macro = f1_score(labels, y_pred, average="macro")
    f1_weighted = f1_score(labels, y_pred, average="weighted")
    report = classification_report(labels, y_pred, output_dict=True)
    cm = confusion_matrix(labels, y_pred, labels=list(unique)).tolist()
    
    # Measure latency for single inference
    clf.fit(X_data, labels)
    t_single_start = time.time()
    for _ in range(50):
        _ = clf.predict_proba([X_data[0]])
    single_latency_ms = (time.time() - t_single_start) / 50 * 1000
    
    results[name] = {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "single_sample_latency_ms": float(single_latency_ms),
        "confusion_matrix": cm,
        "classes": list(unique),
        "report": report
    }
    print(f"[{name}] Acc: {acc*100:.1f}%, F1-macro: {f1_macro:.3f}, Single Inf Latency: {single_latency_ms:.2f}ms")

with open("benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nBenchmark successfully saved to benchmark_results.json")

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

DATA_FILE = "phrases_data.pickle"
MODEL_FILE = "phrase_model_v2.p"

def extract_enhanced_features(seqs):
    """
    Given an array/list of sequences of shape (N, 40, 66) or (40, 66),
    extracts normalized wrist trajectory, scale-normalized landmarks,
    and velocity dynamics.
    Returns feature array of shape (N, 2838) or (1, 2838).
    """
    is_single = False
    seqs_arr = np.asarray(seqs, dtype=np.float32)
    if seqs_arr.ndim == 2:
        seqs_arr = np.expand_dims(seqs_arr, axis=0)
        is_single = True
        
    enhanced_list = []
    for seq in seqs_arr: # seq: (40, 66)
        diffs = np.diff(seq, axis=0) # (39, 66)
        vel_mean = np.mean(diffs, axis=0)
        vel_std = np.std(diffs, axis=0)
        vel_max = np.max(diffs, axis=0)
        
        hand_pts = seq[:, 3:].reshape(len(seq), 21, 3) # (40, 21, 3)
        bbox_diag = np.linalg.norm(hand_pts.max(axis=1) - hand_pts.min(axis=1), axis=1, keepdims=True)
        bbox_diag = np.where(bbox_diag == 0, 1.0, bbox_diag)
        norm_hand = (hand_pts / bbox_diag[:, :, np.newaxis]).reshape(len(seq), 63)
        
        wrist_traj = seq[:, :3] - seq[0, :3]
        norm_seq = np.hstack([wrist_traj, norm_hand]) # (40, 66)
        
        feat_vector = np.concatenate([norm_seq.flatten(), vel_mean, vel_std, vel_max])
        enhanced_list.append(feat_vector)
        
    out = np.asarray(enhanced_list, dtype=np.float32)
    return out

if __name__ == "__main__":
    with open(DATA_FILE, "rb") as f:
        dataset = pickle.load(f)

    sequences = dataset["sequences"]
    labels = dataset["labels"]

    print(f"Loaded {len(sequences)} samples from {DATA_FILE}")
    X_enhanced = extract_enhanced_features(sequences)
    y = np.asarray(labels)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_enhanced, y)

    # Bundle model, metadata, classes, and pipeline spec
    artifact = {
        "model": model,
        "classes": list(model.classes_),
        "version": "2.0.0",
        "feature_type": "normalized_trajectory_and_velocity_dynamics",
        "sequence_length": 40,
        "feature_dim_per_frame": 66,
        "total_features": int(X_enhanced.shape[1]),
        "rejection_threshold": 0.65,
        "stability_window": 5
    }

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(artifact, f)

    print(f"Saved enhanced model artifact to {MODEL_FILE}")
    print(f"Supported classes: {artifact['classes']}")
    print(f"Total features per sample: {artifact['total_features']}")

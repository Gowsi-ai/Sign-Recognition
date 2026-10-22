 HEAD
# SignBridge AI — Real-Time Sign Language Recognition Dashboard

An end-to-end, dynamic sign-language recognition dashboard featuring an enhanced ML model pipeline, scale-invariant temporal feature engineering, prediction stabilization, and rule-based sentence/paragraph synthesis.

---

## Architecture Overview

```
sign-language-detector/
├── phrases_data.pickle                 # Original dataset (backed up safely)
├── phrases_data_original_backup.pickle # Verifiable immutable backup
├── benchmark.py                        # 5-Fold Stratified Cross-Validation benchmark
├── benchmark_results.json              # Measured benchmark scores & latency
├── train_enhanced_model.py             # Feature pipeline & artifact exporter
├── phrase_model_v2.p                   # Validated v2.0.0 model package with metadata
├── inference_engine.py                 # Core reusable ML inference engine & state machine
├── test_pipeline.py                    # Automated unit & integration test suite
├── server.py                           # FastAPI + WebSocket real-time server
└── frontend/                           # React + TypeScript + Vite modern dashboard
    ├── src/App.tsx                     # Dashboard UI, controls & landmark visualizer
    ├── src/index.css                   # Glassmorphic responsive dark mode design
    └── package.json
```

---

## Machine Learning & Hybrid Detection Architecture (v2.5.0)

1. **Instantaneous 3D Landmark Geometric Heuristics**:
   - Computes 3D joint angles, finger extension ratios (Thumb, Index, Middle, Ring, Pinky), and pinch distances in real time for immediate single-frame feedback (<30ms).
   - Fast recognition of static postures: *Thumbs Up (Good/Yes)*, *Peace (V-sign)*, *I Love You*, *OK Sign*, *Pointing (You)*, *Call Me*, *Stop*, etc.

2. **Temporal Dynamic Model & Sequence Resampling**:
   - Normalized wrist trajectory relative to start frame ($t - t_0$).
   - Dynamic interpolation: sequences with $\ge 12$ frames are automatically resampled to 40 frames, eliminating rigid frame waiting.
   - Scale-invariant hand normalization and first-order temporal velocity statistics (2,838 total features).
   - Expanded 14-class lexicon: `Hello`, `Thank you`, `Yes`, `No`, `Please`, `Sorry`, `Good`, `I love you`, `Peace`, `Okay`, `Help`, `You`, `Stop`, `Call me`.

3. **Inference State Machine & Reliability**:
   - **Tracking Flicker Grace Period**: Tolerates momentary hand occlusion or tracking drops (1-5 frames) without wiping sequence progress.
   - **Stability Window**: Requires 3 consecutive identical predictions before confirming.
   - **Cooldown Suppression**: Debounces repeated triggers for 1.4 seconds.

4. **Natural Sentence & Paragraph Synthesis**:
   - Automatically weaves recognized sign tokens into grammatically sound English sentences and flowing conversational paragraphs.
   - Integrated Web Speech Text-to-Speech (TTS), transcript downloading, and real-time interactive gesture reference guide.

---

## Benchmark Results (5-Fold Stratified CV)

| Model Candidate | Feature Set | Accuracy | Macro-F1 | Weighted-F1 | Single Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline Random Forest** | Raw Flattened (2640) | 100.0% | 1.000 | 1.000 | 30.22 ms |
| **Enhanced Random Forest (Selected)** | Normalized + Dynamics (2838) | **100.0%** | **1.000** | **1.000** | 33.89 ms |
| **Enhanced Extra Trees** | Normalized + Dynamics (2838) | 100.0% | 1.000 | 1.000 | 37.21 ms |
| **Enhanced SVM (RBF)** | Normalized + Dynamics (2838) | 100.0% | 1.000 | 1.000 | 0.59 ms |
| **Enhanced KNN** | Normalized + Dynamics (2838) | 100.0% | 1.000 | 1.000 | 50.24 ms |

*Note: While SVM demonstrated low latency, Random Forest v2 with dynamic motion feature extraction was selected for its robust out-of-bag probability calibration and resilience against partial landmark occlusions.*

---

## Running the Application

### 1. Start the FastAPI Backend
```bash
python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Start the Frontend Dashboard
```bash
cd frontend
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

### 3. Run Automated Tests
```bash
python test_pipeline.py
```

# Sign-Recognition
6407f0f862d4afbeccd51e10861b1cb31976b4c4

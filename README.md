# 🤟 SignBridge AI — Sign Language Recognition System

**Bridging the communication gap through Artificial Intelligence.**

SignBridge AI is a machine-learning-based sign language recognition project designed to recognize hand gestures and help make communication more accessible. It uses computer vision to process visual input, extract meaningful hand features, and classify gestures into their corresponding classes.

## 🎯 Project Objective

The primary objective of this project is to explore how Artificial Intelligence (AI), Machine Learning (ML), and Computer Vision can be used to recognize sign language gestures and support communication accessibility.

## ✨ Key Features

- **Hand Gesture Recognition:** Identifies hand gestures from visual input.
- **Computer Vision:** Uses OpenCV for image and video processing.
- **Hand Landmark Detection:** Uses MediaPipe to detect and extract hand landmarks.
- **Feature Extraction:** Converts detected landmarks into numerical features for classification.
- **Machine Learning Classification:** Uses a Random Forest classifier to predict gesture classes.
- **Real-Time Interaction:** Supports live gesture recognition if the webcam inference functionality is enabled in the application.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| OpenCV | Image and video processing |
| MediaPipe | Hand landmark detection |
| Scikit-learn | Machine-learning classification |
| NumPy | Numerical computations |

## ⚙️ How It Works

1. **Input:** Captures a hand gesture through a webcam or processes an available image.
2. **Detection:** MediaPipe identifies hand landmarks.
3. **Feature Extraction:** Converts the detected landmarks into numerical features.
4. **Classification:** The trained Random Forest model predicts the gesture class.
5. **Output:** Displays the predicted gesture in the application.

## 🧠 Machine Learning Approach

The project uses a Random Forest classifier to classify extracted hand features.

Random Forest combines predictions from multiple decision trees. It is a useful choice for structured numerical features and can model nonlinear relationships between features and gesture classes.

**Feature count:** 101 numerical features  
**Gesture classes:** 42 classes  
**Classifier:** Random Forest

*Verify these values against your actual feature-extraction code, trained model, and class-label mapping before publishing.*

## 🚀 Getting Started

### Prerequisites

- Python installed on your system
- A webcam, if using live recognition
- Git
- The required Python packages and trained model files

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Gowsi-ai/Sign-Recognition.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Sign-Recognition
   ```

3. Create and activate a virtual environment:

   **Windows PowerShell**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

4. Install the dependencies listed in your project’s `requirements.txt`, if available:

   ```bash
   pip install -r requirements.txt
   ```

5. Run the application using the correct entry-point script in your repository. For example, if `app.py` is the entry point:

   ```bash
   python app.py
   ```

*Note: Confirm the dependency file, entry-point filename, model paths, and installation instructions match the current repository.*

## 📁 Project Structure

The structure below is illustrative. Update it to match the files in your repository.

```text
Sign-Recognition/
├── app.py
├── function.py
├── README.md
├── requirements.txt
├── model.p
├── phrase_model.p
├── phrase_model_v2.p
├── dataset/
├── data/
└── frontend/
```

## 📊 Results and Evaluation

The system is designed to classify hand gestures using extracted numerical features.

For a complete evaluation, include:
- Test-set accuracy
- Precision, recall, and F1-score
- Gesture classes correctly recognized
- Real-time inference performance, if measured

*Add actual measured results here rather than reporting unverified accuracy figures.*

## 🔮 Future Enhancements

- Improve recognition accuracy under different lighting conditions.
- Expand the supported vocabulary of gestures.
- Improve handling of different hand positions and backgrounds.
- Add sentence-level or continuous sign recognition, if not already supported.
- Develop an accessible interface for users.

## 👥 Contributors

Add the names of the project team members and describe each person's contribution.

## 📌 Project Status

Developed as an AI and Machine Learning project. Features and supported gestures depend on the current implementation.

## 📄 License

Add a license appropriate for your project and confirm that you have permission to redistribute any included datasets and model files.

---

⭐ If you find this project interesting, consider starring the repository!

**Repository:** https://github.com/Gowsi-ai/Sign-Recognition

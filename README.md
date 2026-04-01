# ASL Sign Language Recognition

Real-time American Sign Language (ASL) alphabet recognition using hand landmarks.

## 🎯 Features

- **Real-time ASL Recognition**: Recognizes A-Z hand signs via webcam
- **Hand Landmark Detection**: Uses MediaPipe for robust hand tracking
- **High Accuracy**: ~97% accuracy on test set
- **Fast Inference**: Lightweight Random Forest classifier
- **Visual Feedback**: Hand skeleton overlay and confidence display

## 🚀 Quick Start

### 1. Create Virtual Environment

From the project root:

```bash
python -m venv .venv
```

### 2. Activate `.venv`

PowerShell (Windows):

```powershell
.\.venv\Scripts\Activate.ps1
```

Command Prompt (Windows):

```bat
.venv\Scripts\activate.bat
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python app.py
```

### 5. Deactivate (Optional)

```bash
deactivate
```

### Controls

- `q` - Quit application
- `s` - Toggle skeleton display
- `c` - Clear current text
- `Backspace` - Delete last character

## 📁 Project Structure

```
Sign Language Recognition/
├── app.py                      # 🎯 Main entry point - RUN THIS
├── src/
│   ├── config.py               # Configuration settings
│   └── train.py                # Training script
├── models/
│   ├── landmark_classifier.pkl # Trained classifier
│   └── landmark_scaler.pkl     # Feature scaler
├── landmark_data/
│   ├── hand_landmarker.task    # MediaPipe model (auto-downloaded)
│   └── landmark_dataset.npz    # Extracted landmark features
├── asl_alphabet_train/         # Training dataset (A-Z folders)
├── asl_alphabet_test/          # Test dataset
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🔧 Retraining

To retrain the model with different settings:

```bash
python src/train.py
```

This will:

1. Extract hand landmarks from the ASL dataset
2. Train Random Forest and MLP classifiers
3. Save the best model

## 💡 How It Works

### Why Landmarks Instead of Images?

Traditional image-based CNNs struggle with:

- Different backgrounds
- Varying lighting conditions
- Different skin tones
- Camera quality differences

Our landmark-based approach:

1. **Detects** 21 hand keypoints using MediaPipe
2. **Normalizes** positions relative to wrist (translation invariant)
3. **Scales** by hand size (scale invariant)
4. **Classifies** using only 42 geometric features

This makes recognition robust across different environments.

### Model Architecture

- **Hand Detection**: MediaPipe Hand Landmarker
- **Feature Extraction**: 21 landmarks × 2 coordinates = 42 features
- **Classifier**: Random Forest (200 trees)
- **Accuracy**: ~97% on 26 classes (A-Z)

## 📊 Performance

| Metric        | Value    |
| ------------- | -------- |
| Classes       | 26 (A-Z) |
| Test Accuracy | ~97%     |
| FPS           | 15-30    |
| Model Size    | ~200 KB  |

## 📋 Requirements

- Python 3.8+
- Webcam
- Windows/Linux/macOS

## 📝 Notes

- **J and Z**: These letters involve motion, so static recognition may be less accurate
- **Lighting**: Works best with good lighting
- **Hand Position**: Keep your hand within the camera frame

## 👥 Author

Academic Project - 2024

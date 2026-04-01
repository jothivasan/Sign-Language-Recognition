"""
ASL Sign Language Recognition - Training Script

Extracts hand landmarks from the ASL Alphabet dataset and trains a classifier.

Usage: python src/train.py

Author: Academic Project
Date: 2024
"""

import os
import sys
import numpy as np
import pickle
from datetime import datetime

# Add packages to path
MEDIAPIPE_PKG = os.path.join(os.environ.get('USERPROFILE', os.path.expanduser('~')), 'mediapipe_pkg')
TENSORFLOW_PKG = os.path.join(os.environ.get('USERPROFILE', os.path.expanduser('~')), 'tensorflow_pkg')
for pkg in [MEDIAPIPE_PKG, TENSORFLOW_PKG]:
    if pkg not in sys.path and os.path.exists(pkg):
        sys.path.insert(0, pkg)

from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Project imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from config import (
    TRAIN_DATA_PATH, LANDMARK_MODEL_PATH, SCALER_PATH,
    LANDMARK_DATASET_PATH, HAND_LANDMARKER_PATH, LANDMARK_DATA_DIR,
    CLASS_LABELS, MAX_SAMPLES_PER_CLASS, TEST_SPLIT, RANDOM_SEED, MODEL_URL
)


# =============================================================================
# MODEL DOWNLOAD
# =============================================================================

def download_hand_model():
    """Download MediaPipe hand landmarker model."""
    if os.path.exists(HAND_LANDMARKER_PATH):
        return True
    
    print("📥 Downloading hand landmarker model...")
    try:
        import urllib.request
        os.makedirs(LANDMARK_DATA_DIR, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, HAND_LANDMARKER_PATH)
        print("✅ Downloaded")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


# =============================================================================
# LANDMARK EXTRACTION
# =============================================================================

def extract_landmarks_from_dataset():
    """Extract hand landmarks from all images in the dataset."""
    print("\n" + "=" * 60)
    print("🔍 EXTRACTING LANDMARKS FROM DATASET")
    print("=" * 60)
    
    if not download_hand_model():
        return None, None
    
    # Create detector
    base_options = python.BaseOptions(model_asset_path=HAND_LANDMARKER_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)
    
    np.random.seed(RANDOM_SEED)
    
    all_features = []
    all_labels = []
    
    print(f"📂 Dataset: {TRAIN_DATA_PATH}")
    print(f"📊 Samples/class: {MAX_SAMPLES_PER_CLASS}\n")
    
    for class_idx, class_name in enumerate(CLASS_LABELS):
        class_path = os.path.join(TRAIN_DATA_PATH, class_name)
        
        if not os.path.exists(class_path):
            print(f"⚠️ Missing: {class_name}")
            continue
        
        # Get images
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(images) > MAX_SAMPLES_PER_CLASS:
            images = list(np.random.choice(images, MAX_SAMPLES_PER_CLASS, replace=False))
        
        class_features = []
        for img_file in tqdm(images, desc=f"Class {class_name}", leave=False):
            try:
                img = Image.open(os.path.join(class_path, img_file)).convert('RGB')
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(img))
                result = detector.detect(mp_image)
                
                if result.hand_landmarks:
                    landmarks = np.array([[lm.x, lm.y] for lm in result.hand_landmarks[0]])
                    wrist = landmarks[0]
                    normalized = landmarks - wrist
                    scale = np.linalg.norm(landmarks[9] - landmarks[0])
                    if scale > 0:
                        normalized = normalized / scale
                    class_features.append(normalized.flatten())
            except:
                pass
        
        all_features.extend(class_features)
        all_labels.extend([class_idx] * len(class_features))
        print(f"   {class_name}: {len(class_features)} samples")
    
    detector.close()
    
    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    
    print(f"\n📊 Total: {len(X)} samples, {len(set(y))} classes")
    
    # Save dataset
    np.savez(LANDMARK_DATASET_PATH, X=X, y=y, class_labels=CLASS_LABELS)
    print(f"💾 Saved: {LANDMARK_DATASET_PATH}")
    
    return X, y


def load_or_extract_dataset():
    """Load existing dataset or extract new one."""
    if os.path.exists(LANDMARK_DATASET_PATH):
        print(f"📂 Loading: {LANDMARK_DATASET_PATH}")
        data = np.load(LANDMARK_DATASET_PATH, allow_pickle=True)
        return data['X'], data['y']
    return extract_landmarks_from_dataset()


# =============================================================================
# TRAINING
# =============================================================================

def train_classifiers(X_train, y_train, X_test, y_test):
    """Train and compare classifiers."""
    print("\n" + "=" * 60)
    print("🎯 TRAINING CLASSIFIERS")
    print("=" * 60)
    
    results = {}
    
    # Random Forest
    print("\n🌲 Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced_subsample',
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_acc = rf.score(X_test, y_test)
    results['RandomForest'] = (rf, rf_acc)
    print(f"   Accuracy: {rf_acc:.2%}")

    # Extra Trees
    print("\n🌳 Extra Trees...")
    et = ExtraTreesClassifier(
        n_estimators=700,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    et.fit(X_train, y_train)
    et_acc = et.score(X_test, y_test)
    results['ExtraTrees'] = (et, et_acc)
    print(f"   Accuracy: {et_acc:.2%}")
    
    # MLP
    print("\n🧠 MLP Neural Network...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        learning_rate_init=0.001,
        alpha=1e-4,
        max_iter=700,
        n_iter_no_change=20,
        early_stopping=True,
        random_state=RANDOM_SEED
    )
    mlp.fit(X_train, y_train)
    mlp_acc = mlp.score(X_test, y_test)
    results['MLP'] = (mlp, mlp_acc)
    print(f"   Accuracy: {mlp_acc:.2%}")
    
    # Select best
    best_name = max(results, key=lambda k: results[k][1])
    best_model, best_acc = results[best_name]
    
    print(f"\n🏆 Best: {best_name} ({best_acc:.2%})")
    
    return best_model, best_name, best_acc


def main():
    """Main training function."""
    print("\n" + "=" * 60)
    print("🚀 ASL LANDMARK CLASSIFIER TRAINING")
    print("=" * 60)
    
    start = datetime.now()
    
    # Load/extract data
    X, y = load_or_extract_dataset()
    if X is None:
        return
    
    # Scale features
    print("\n⚙️ Preparing data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y
    )
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train
    model, model_name, accuracy = train_classifiers(X_train, y_train, X_test, y_test)
    
    # Save
    print("\n💾 Saving model...")
    with open(LANDMARK_MODEL_PATH, 'wb') as f:
        pickle.dump({'model': model, 'model_name': model_name, 'accuracy': accuracy, 'class_labels': CLASS_LABELS}, f)
    
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"   Model: {LANDMARK_MODEL_PATH}")
    print(f"   Scaler: {SCALER_PATH}")
    
    # Summary
    duration = datetime.now() - start
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE")
    print("=" * 60)
    print(f"   Duration: {duration}")
    print(f"   Model: {model_name}")
    print(f"   Accuracy: {accuracy:.2%}")
    print(f"\n🎯 Run: python app.py")


if __name__ == "__main__":
    main()

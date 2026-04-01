"""
ASL Sign Language Recognition - Configuration

This module contains all configuration settings for the
hand landmark-based sign language recognition system.

Author: Academic Project
Date: 2024
"""

import os

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, "asl_alphabet_train", "asl_alphabet_train")

# Model paths
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LANDMARK_MODEL_PATH = os.path.join(MODELS_DIR, "landmark_classifier.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "landmark_scaler.pkl")

# Landmark data
LANDMARK_DATA_DIR = os.path.join(PROJECT_ROOT, "landmark_data")
LANDMARK_DATASET_PATH = os.path.join(LANDMARK_DATA_DIR, "landmark_dataset.npz")
HAND_LANDMARKER_PATH = os.path.join(LANDMARK_DATA_DIR, "hand_landmarker.task")

# Create directories
for dir_path in [MODELS_DIR, LANDMARK_DATA_DIR]:
    os.makedirs(dir_path, exist_ok=True)


# =============================================================================
# ASL ALPHABET CLASSES
# =============================================================================

CLASS_LABELS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]

NUM_CLASSES = len(CLASS_LABELS)


# =============================================================================
# MEDIAPIPE SETTINGS
# =============================================================================

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

NUM_LANDMARKS = 21
FEATURES_PER_LANDMARK = 2
TOTAL_FEATURES = NUM_LANDMARKS * FEATURES_PER_LANDMARK  # 42


# =============================================================================
# CAMERA SETTINGS
# =============================================================================

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


# =============================================================================
# TRAINING SETTINGS
# =============================================================================

MAX_SAMPLES_PER_CLASS = 1000
TEST_SPLIT = 0.2
RANDOM_SEED = 42


# =============================================================================
# PREDICTION SETTINGS
# =============================================================================

CONFIDENCE_THRESHOLD = 0.5
PREDICTION_SMOOTHING_WINDOW = 5


# =============================================================================
# DISPLAY COLORS (BGR)
# =============================================================================

TEXT_COLOR = (0, 255, 0)       # Green
WARNING_COLOR = (0, 165, 255)  # Orange

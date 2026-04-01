"""
ASL Sign Language Recognition - Enhanced Word Builder Application

Real-time sign language alphabet recognition with word building capabilities.
This application extends the base recognition system with:
- START button to control camera activation
- Hold-based letter confirmation (2 seconds)
- Word formation from confirmed letters
- Space gesture support for word separation
- Clean and intuitive UI

Usage: python app.py

Controls:
- Click 'START' button to begin
- Hold a letter sign for 2 seconds to confirm
- Use open palm (5 fingers spread) for SPACE
- Press 'q' to quit
- Press 's' to toggle skeleton display
- Press 'c' to clear the current word
- Press 'Backspace' to delete last letter

Author: Academic Project
Date: 2024
"""

import os
import sys
import cv2
import numpy as np
import pickle
import time
from collections import deque, Counter

# Add packages to path
MEDIAPIPE_PKG = os.path.join(os.environ.get('USERPROFILE', os.path.expanduser('~')), 'mediapipe_pkg')
TENSORFLOW_PKG = os.path.join(os.environ.get('USERPROFILE', os.path.expanduser('~')), 'tensorflow_pkg')
for pkg in [MEDIAPIPE_PKG, TENSORFLOW_PKG]:
    if pkg not in sys.path and os.path.exists(pkg):
        sys.path.insert(0, pkg)

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Project imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from src.config import (
    LANDMARK_MODEL_PATH, SCALER_PATH, HAND_LANDMARKER_PATH,
    CLASS_LABELS, CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT,
    CONFIDENCE_THRESHOLD, PREDICTION_SMOOTHING_WINDOW,
    TEXT_COLOR, WARNING_COLOR, MODEL_URL, LANDMARK_DATA_DIR
)


# =============================================================================
# CONFIGURATION FOR WORD BUILDER
# =============================================================================

HOLD_TIME_SECONDS = 2.0  # Time to hold a letter before confirmation
MIN_TIME_BETWEEN_CAPTURES = 0.5  # Minimum time between letter captures
SPACE_CONFIDENCE_THRESHOLD = 0.6  # Confidence threshold for space detection


# =============================================================================
# UI COLORS (BGR format)
# =============================================================================

class Colors:
    """Color palette for the UI."""
    BACKGROUND = (30, 30, 30)
    PANEL_BG = (45, 45, 45)
    PANEL_BORDER = (60, 60, 60)
    
    # Button colors
    BUTTON_START = (0, 180, 100)
    BUTTON_START_HOVER = (0, 200, 120)
    BUTTON_STOP = (60, 60, 200)
    BUTTON_STOP_HOVER = (80, 80, 220)
    BUTTON_TEXT = (255, 255, 255)
    
    # Text colors
    TEXT_PRIMARY = (255, 255, 255)
    TEXT_SECONDARY = (180, 180, 180)
    TEXT_MUTED = (120, 120, 120)
    TEXT_SUCCESS = (100, 255, 150)
    TEXT_WARNING = (100, 200, 255)
    TEXT_PREVIEW = (255, 200, 100)
    
    # Progress bar
    PROGRESS_BG = (60, 60, 60)
    PROGRESS_FILL = (100, 255, 150)
    PROGRESS_FILL_SPACE = (255, 200, 100)
    
    # Word display
    WORD_BG = (50, 50, 50)
    WORD_TEXT = (255, 255, 255)
    CONFIRMED_LETTER = (100, 255, 150)


# =============================================================================
# HAND CONNECTIONS FOR SKELETON DRAWING
# =============================================================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)             # Palm
]


# =============================================================================
# UTILITY FUNCTIONS (Same as original)
# =============================================================================

def download_hand_model():
    """Download MediaPipe hand landmarker model if not present."""
    if os.path.exists(HAND_LANDMARKER_PATH):
        return True
    
    print("📥 Downloading hand landmarker model...")
    try:
        import urllib.request
        os.makedirs(LANDMARK_DATA_DIR, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, HAND_LANDMARKER_PATH)
        print("✅ Model downloaded")
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def load_classifier():
    """Load the trained landmark classifier and scaler."""
    if not os.path.exists(LANDMARK_MODEL_PATH):
        print(f"❌ Classifier not found: {LANDMARK_MODEL_PATH}")
        print("   Run: python src/train.py")
        return None, None
    
    if not os.path.exists(SCALER_PATH):
        print(f"❌ Scaler not found: {SCALER_PATH}")
        return None, None
    
    with open(LANDMARK_MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"✅ Model loaded: {model_data['model_name']} ({model_data['accuracy']:.1%} accuracy)")
    return model_data['model'], scaler


def create_hand_detector():
    """Initialize MediaPipe hand landmarker."""
    if not download_hand_model():
        return None
    
    base_options = python.BaseOptions(model_asset_path=HAND_LANDMARKER_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return vision.HandLandmarker.create_from_options(options)


# =============================================================================
# LANDMARK PROCESSING (Same as original)
# =============================================================================

def extract_features(hand_landmarks, frame_shape):
    """Extract normalized features and pixel coordinates from landmarks."""
    h, w = frame_shape[:2]
    
    landmarks = []
    pixel_points = []
    
    for lm in hand_landmarks:
        landmarks.append([lm.x, lm.y])
        pixel_points.append((int(lm.x * w), int(lm.y * h)))
    
    landmarks = np.array(landmarks)
    
    # Normalize: translation (relative to wrist) and scale invariance
    wrist = landmarks[0]
    normalized = landmarks - wrist
    scale = np.linalg.norm(landmarks[9] - landmarks[0])
    if scale > 0:
        normalized = normalized / scale
    
    return normalized.flatten(), pixel_points


def predict(model, scaler, features):
    """Make prediction from landmark features."""
    scaled = scaler.transform(features.reshape(1, -1))
    
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(scaled)[0]
        idx = np.argmax(probs)
        return CLASS_LABELS[idx], probs[idx], probs
    else:
        idx = model.predict(scaled)[0]
        return CLASS_LABELS[idx], 1.0, None


# =============================================================================
# SPACE GESTURE DETECTION
# =============================================================================

def is_space_gesture(hand_landmarks):
    """
    Detect open palm gesture for SPACE.
    Open palm = all fingers extended and spread apart.
    """
    if not hand_landmarks:
        return False, 0.0
    
    # Get landmark positions
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    
    # Finger tips: 4 (thumb), 8 (index), 12 (middle), 16 (ring), 20 (pinky)
    # Finger MCPs: 2 (thumb), 5 (index), 9 (middle), 13 (ring), 17 (pinky)
    
    tips = [4, 8, 12, 16, 20]
    mcps = [2, 5, 9, 13, 17]
    pips = [3, 6, 10, 14, 18]
    
    # Check if all fingers are extended (tip above pip in y for non-thumb)
    fingers_extended = 0
    
    # Thumb check (x-direction based)
    if abs(landmarks[4][0] - landmarks[2][0]) > 0.04:
        fingers_extended += 1
    
    # Other fingers check (y-direction based)
    for tip, pip in zip(tips[1:], pips[1:]):
        if landmarks[tip][1] < landmarks[pip][1]:  # tip above pip
            fingers_extended += 1
    
    # Check for spread (distance between finger tips)
    if fingers_extended >= 4:
        # Calculate average distance between adjacent fingertips
        spread = 0
        for i in range(len(tips) - 1):
            dist = np.linalg.norm(landmarks[tips[i]][:2] - landmarks[tips[i+1]][:2])
            spread += dist
        spread /= 4
        
        # If fingers are extended and spread, it's a space gesture
        if spread > 0.08:
            confidence = min(1.0, (fingers_extended / 5.0) * (spread / 0.12))
            return True, confidence
    
    return False, 0.0


# =============================================================================
# PREDICTION SMOOTHER (Same as original)
# =============================================================================

class PredictionSmoother:
    """Smooth predictions over multiple frames to reduce jitter."""
    
    def __init__(self, window_size=PREDICTION_SMOOTHING_WINDOW):
        self.predictions = deque(maxlen=window_size)
        self.confidences = deque(maxlen=window_size)
    
    def add(self, prediction, confidence):
        self.predictions.append(prediction)
        self.confidences.append(confidence)
    
    def get(self):
        if not self.predictions:
            return None, 0.0
        
        counts = Counter(self.predictions)
        best = counts.most_common(1)[0]
        pred, count = best
        
        confs = [c for p, c in zip(self.predictions, self.confidences) if p == pred]
        avg_conf = np.mean(confs) * (0.8 + 0.2 * count / len(self.predictions))
        
        return pred, avg_conf
    
    def clear(self):
        self.predictions.clear()
        self.confidences.clear()


# =============================================================================
# LETTER HOLD TIMER
# =============================================================================

class LetterHoldTimer:
    """
    Manages the hold-based letter confirmation system.
    A letter must be held steady for HOLD_TIME_SECONDS to be confirmed.
    """
    
    def __init__(self, hold_time=HOLD_TIME_SECONDS, min_gap=MIN_TIME_BETWEEN_CAPTURES):
        self.hold_time = hold_time
        self.min_gap = min_gap
        
        self.current_letter = None
        self.hold_start_time = None
        self.last_capture_time = 0
        self.is_space_mode = False
    
    def update(self, letter, is_space=False):
        """
        Update the timer with the current detected letter.
        Returns (confirmed_letter, progress) where:
        - confirmed_letter is the letter if hold time is complete, else None
        - progress is 0.0 to 1.0 indicating hold progress
        """
        current_time = time.time()
        
        # Check if letter changed
        if letter != self.current_letter or is_space != self.is_space_mode:
            self.current_letter = letter
            self.hold_start_time = current_time
            self.is_space_mode = is_space
        
        # Calculate progress
        if self.hold_start_time is None:
            return None, 0.0
        
        elapsed = current_time - self.hold_start_time
        progress = min(1.0, elapsed / self.hold_time)
        
        # Check if hold time completed and minimum gap passed
        if progress >= 1.0 and (current_time - self.last_capture_time) >= self.min_gap:
            confirmed = self.current_letter
            self.last_capture_time = current_time
            self.hold_start_time = current_time  # Reset for next capture
            return confirmed, 1.0
        
        return None, progress
    
    def reset(self):
        """Reset the timer."""
        self.current_letter = None
        self.hold_start_time = None
        self.is_space_mode = False


# =============================================================================
# WORD BUILDER
# =============================================================================

class WordBuilder:
    """Manages the word/sentence being built from confirmed letters."""
    
    def __init__(self):
        self.text = ""
        self.last_confirmed = None
        self.last_confirmed_time = 0
    
    def add_letter(self, letter):
        """Add a confirmed letter to the word."""
        self.text += letter
        self.last_confirmed = letter
        self.last_confirmed_time = time.time()
    
    def add_space(self):
        """Add a space to the word."""
        if self.text and not self.text.endswith(' '):
            self.text += ' '
            self.last_confirmed = '[SPACE]'
            self.last_confirmed_time = time.time()
    
    def backspace(self):
        """Remove the last character."""
        if self.text:
            self.text = self.text[:-1]
    
    def clear(self):
        """Clear the entire text."""
        self.text = ""
        self.last_confirmed = None
    
    def get_text(self):
        """Get the current text."""
        return self.text
    
    def get_last_confirmed(self):
        """Get the last confirmed letter with fade effect."""
        if self.last_confirmed is None:
            return None
        
        # Show confirmation for 1.5 seconds
        if time.time() - self.last_confirmed_time > 1.5:
            return None
        return self.last_confirmed


# =============================================================================
# UI COMPONENTS
# =============================================================================

class Button:
    """A clickable button for the UI."""
    
    def __init__(self, x, y, width, height, text, color, hover_color, text_color):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.is_hovered = False
    
    def contains_point(self, px, py):
        """Check if a point is inside the button."""
        return (self.x <= px <= self.x + self.width and 
                self.y <= py <= self.y + self.height)
    
    def draw(self, frame):
        """Draw the button on the frame."""
        color = self.hover_color if self.is_hovered else self.color
        
        # Draw button background with rounded corners effect
        cv2.rectangle(frame, (self.x, self.y), 
                     (self.x + self.width, self.y + self.height), 
                     color, -1)
        
        # Draw border
        cv2.rectangle(frame, (self.x, self.y), 
                     (self.x + self.width, self.y + self.height), 
                     tuple(min(255, c + 30) for c in color), 2)
        
        # Draw text centered
        text_size = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = self.x + (self.width - text_size[0]) // 2
        text_y = self.y + (self.height + text_size[1]) // 2
        cv2.putText(frame, self.text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.text_color, 2)


def draw_rounded_rect(frame, x, y, w, h, color, radius=10):
    """Draw a rectangle with rounded corners."""
    # Main rectangle
    cv2.rectangle(frame, (x + radius, y), (x + w - radius, y + h), color, -1)
    cv2.rectangle(frame, (x, y + radius), (x + w, y + h - radius), color, -1)
    # Corners
    cv2.circle(frame, (x + radius, y + radius), radius, color, -1)
    cv2.circle(frame, (x + w - radius, y + radius), radius, color, -1)
    cv2.circle(frame, (x + radius, y + h - radius), radius, color, -1)
    cv2.circle(frame, (x + w - radius, y + h - radius), radius, color, -1)


def draw_progress_bar(frame, x, y, width, height, progress, bg_color, fill_color):
    """Draw a progress bar."""
    # Background
    cv2.rectangle(frame, (x, y), (x + width, y + height), bg_color, -1)
    
    # Fill
    fill_width = int(width * progress)
    if fill_width > 0:
        cv2.rectangle(frame, (x, y), (x + fill_width, y + height), fill_color, -1)
    
    # Border
    cv2.rectangle(frame, (x, y), (x + width, y + height), (80, 80, 80), 1)


# =============================================================================
# VISUALIZATION
# =============================================================================

def draw_skeleton(frame, points, show=True):
    """Draw hand skeleton on frame."""
    if not points or not show:
        return frame
    
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 255, 255), 2)
    
    for i, pt in enumerate(points):
        color = (0, 0, 255) if i in [4, 8, 12, 16, 20] else (255, 0, 0) if i == 0 else (0, 255, 0)
        radius = 8 if i in [0, 4, 8, 12, 16, 20] else 5
        cv2.circle(frame, pt, radius, color, -1)
    
    return frame


def draw_start_screen(frame, button):
    """Draw the start screen with instructions."""
    h, w = frame.shape[:2]
    
    # Dark overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), Colors.BACKGROUND, -1)
    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
    
    # Title
    title = "ASL Sign Language Recognition"
    title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
    cv2.putText(frame, title, ((w - title_size[0]) // 2, h // 4), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, Colors.TEXT_PRIMARY, 2)
    
    # Subtitle
    subtitle = "Word Builder Mode"
    sub_size = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.putText(frame, subtitle, ((w - sub_size[0]) // 2, h // 4 + 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, Colors.TEXT_SUCCESS, 2)
    
    # Instructions
    instructions = [
        "Instructions:",
        "- Hold a letter sign for 2 seconds to confirm",
        "- Open palm (5 fingers spread) = SPACE",
        "- Press 'C' to clear text",
        "- Press 'Backspace' to delete last letter",
        "- Press 'Q' to quit"
    ]
    
    y_start = h // 2 - 40
    for i, text in enumerate(instructions):
        color = Colors.TEXT_WARNING if i == 0 else Colors.TEXT_SECONDARY
        size = 0.6 if i == 0 else 0.5
        cv2.putText(frame, text, (w // 4, y_start + i * 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, size, color, 1)
    
    # Draw the start button
    button.draw(frame)
    
    return frame


def draw_main_ui(frame, prediction, confidence, fps, hand_detected, 
                 hold_progress, is_space, word_builder, show_skeleton_status):
    """Draw the main application UI overlay."""
    h, w = frame.shape[:2]
    
    # ==== Top Panel: Current Detection ====
    panel_height = 120
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_height), Colors.PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.line(frame, (0, panel_height), (w, panel_height), Colors.PANEL_BORDER, 2)
    
    # Left side: Live preview
    cv2.putText(frame, "LIVE PREVIEW", (20, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.TEXT_MUTED, 1)
    
    if hand_detected:
        if is_space:
            preview_text = "[SPACE]"
            preview_color = Colors.TEXT_WARNING
        elif prediction and confidence >= CONFIDENCE_THRESHOLD:
            preview_text = prediction
            preview_color = Colors.TEXT_PREVIEW
        else:
            preview_text = "?"
            preview_color = Colors.TEXT_MUTED
        
        cv2.putText(frame, preview_text, (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, preview_color, 3)
        
        # Confidence
        conf_text = f"Conf: {confidence:.0%}" if not is_space else "Gesture"
        cv2.putText(frame, conf_text, (20, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.TEXT_SECONDARY, 1)
    else:
        cv2.putText(frame, "No Hand", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, Colors.TEXT_MUTED, 2)
        cv2.putText(frame, "Show your hand", (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.TEXT_MUTED, 1)
    
    # Center: Hold progress
    progress_x = w // 2 - 100
    cv2.putText(frame, "HOLD PROGRESS", (progress_x, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.TEXT_MUTED, 1)
    
    fill_color = Colors.PROGRESS_FILL_SPACE if is_space else Colors.PROGRESS_FILL
    draw_progress_bar(frame, progress_x, 40, 200, 25, hold_progress, 
                     Colors.PROGRESS_BG, fill_color)
    
    progress_pct = f"{hold_progress * 100:.0f}%"
    cv2.putText(frame, progress_pct, (progress_x + 85, 58), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.TEXT_PRIMARY, 1)
    
    hold_text = "Hold 2s to confirm"
    cv2.putText(frame, hold_text, (progress_x + 30, 85), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, Colors.TEXT_SECONDARY, 1)
    
    # Right side: Last confirmed
    last_confirmed = word_builder.get_last_confirmed()
    cv2.putText(frame, "CONFIRMED", (w - 150, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.TEXT_MUTED, 1)
    
    if last_confirmed:
        cv2.putText(frame, last_confirmed, (w - 150, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.8, Colors.CONFIRMED_LETTER, 3)
        cv2.putText(frame, "Added!", (w - 140, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.TEXT_SUCCESS, 1)
    else:
        cv2.putText(frame, "-", (w - 120, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, Colors.TEXT_MUTED, 2)
    
    # ==== Bottom Panel: Word Display ====
    bottom_panel_height = 80
    bottom_y = h - bottom_panel_height
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, bottom_y), (w, h), Colors.WORD_BG, -1)
    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
    cv2.line(frame, (0, bottom_y), (w, bottom_y), Colors.PANEL_BORDER, 2)
    
    cv2.putText(frame, "FORMED WORD:", (20, bottom_y + 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.TEXT_MUTED, 1)
    
    # Word text with cursor
    word_text = word_builder.get_text()
    if len(word_text) > 40:
        word_text = "..." + word_text[-37:]
    
    cursor = "_" if int(time.time() * 2) % 2 == 0 else ""
    display_text = word_text + cursor
    
    cv2.putText(frame, display_text, (20, bottom_y + 55), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, Colors.WORD_TEXT, 2)
    
    # Character count
    char_count = f"Chars: {len(word_builder.get_text())}"
    cv2.putText(frame, char_count, (w - 100, bottom_y + 55), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.TEXT_MUTED, 1)
    
    # ==== Status bar ====
    status = "Hand ✓" if hand_detected else "No Hand"
    skeleton_status = "Skeleton: ON" if show_skeleton_status else "Skeleton: OFF"
    cv2.putText(frame, f"FPS: {fps:.0f} | {status} | {skeleton_status}", 
               (10, bottom_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, Colors.TEXT_MUTED, 1)
    
    cv2.putText(frame, "'Q' quit | 'S' skeleton | 'C' clear | 'Backspace' delete", 
               (w - 350, bottom_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, Colors.TEXT_MUTED, 1)
    
    return frame


# =============================================================================
# MOUSE CALLBACK
# =============================================================================

mouse_x, mouse_y = 0, 0
mouse_clicked = False

def mouse_callback(event, x, y, flags, param):
    """Handle mouse events."""
    global mouse_x, mouse_y, mouse_clicked
    mouse_x, mouse_y = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_clicked = True


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    global mouse_clicked
    
    print("\n" + "=" * 60)
    print("🤟 ASL SIGN LANGUAGE RECOGNITION - WORD BUILDER")
    print("=" * 60)
    
    # Load classifier
    model, scaler = load_classifier()
    if model is None:
        return
    
    # Initialize MediaPipe
    print("⚙️ Initializing hand detector...")
    detector = create_hand_detector()
    if detector is None:
        return
    print("✅ Hand detector ready")
    
    # Initialize components
    smoother = PredictionSmoother()
    hold_timer = LetterHoldTimer()
    word_builder = WordBuilder()
    
    # State
    camera_active = False
    cap = None
    show_skeleton = True
    fps = 0
    fps_time = time.time()
    fps_count = 0
    timestamp = 0
    
    # Create window
    window = "ASL Sign Language Recognition - Word Builder"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, CAMERA_WIDTH, CAMERA_HEIGHT)
    cv2.setMouseCallback(window, mouse_callback)
    
    # Create start button
    btn_width, btn_height = 200, 60
    start_button = Button(
        (CAMERA_WIDTH - btn_width) // 2,
        int(CAMERA_HEIGHT * 0.7),
        btn_width, btn_height,
        "START",
        Colors.BUTTON_START,
        Colors.BUTTON_START_HOVER,
        Colors.BUTTON_TEXT
    )
    
    print("\n" + "=" * 60)
    print("▶️ Click START to begin!")
    print("=" * 60 + "\n")
    
    try:
        while True:
            if not camera_active:
                # Show start screen
                frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
                frame[:] = Colors.BACKGROUND
                
                # Update button hover state
                start_button.is_hovered = start_button.contains_point(mouse_x, mouse_y)
                
                # Check for button click
                if mouse_clicked and start_button.contains_point(mouse_x, mouse_y):
                    print("📹 Starting camera...")
                    cap = cv2.VideoCapture(CAMERA_INDEX)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                        camera_active = True
                        print("✅ Camera started!")
                    else:
                        print("❌ Cannot open camera!")
                    mouse_clicked = False
                
                frame = draw_start_screen(frame, start_button)
                mouse_clicked = False
                
            else:
                # Main recognition loop
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame = cv2.flip(frame, 1)  # Mirror
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                
                timestamp += 33
                result = detector.detect_for_video(mp_image, timestamp)
                
                prediction = None
                confidence = 0.0
                points = None
                hand_detected = False
                is_space = False
                hold_progress = 0.0
                
                if result.hand_landmarks:
                    hand_detected = True
                    hand_lms = result.hand_landmarks[0]
                    features, points = extract_features(hand_lms, frame.shape)
                    
                    # Check for space gesture first
                    is_space, space_conf = is_space_gesture(hand_lms)
                    
                    if is_space and space_conf >= SPACE_CONFIDENCE_THRESHOLD:
                        # Space gesture detected
                        confirmed, hold_progress = hold_timer.update("[SPACE]", is_space=True)
                        if confirmed:
                            word_builder.add_space()
                            print(f"Added: [SPACE] | Text: '{word_builder.get_text()}'")
                        smoother.clear()
                        prediction = "[SPACE]"
                        confidence = space_conf
                    else:
                        # Regular letter prediction
                        pred, conf, probs = predict(model, scaler, features)
                        smoother.add(pred, conf)
                        prediction, confidence = smoother.get()
                        
                        if prediction and confidence >= CONFIDENCE_THRESHOLD:
                            confirmed, hold_progress = hold_timer.update(prediction, is_space=False)
                            if confirmed:
                                word_builder.add_letter(confirmed)
                                print(f"Added: {confirmed} | Text: '{word_builder.get_text()}'")
                        else:
                            hold_timer.reset()
                    
                    # Draw skeleton
                    frame = draw_skeleton(frame, points, show_skeleton)
                else:
                    smoother.clear()
                    hold_timer.reset()
                
                # FPS calculation
                fps_count += 1
                if time.time() - fps_time >= 0.5:
                    fps = fps_count / (time.time() - fps_time)
                    fps_count = 0
                    fps_time = time.time()
                
                # Draw UI
                frame = draw_main_ui(frame, prediction, confidence, fps, hand_detected,
                                    hold_progress, is_space, word_builder, show_skeleton)
            
            # Show frame
            cv2.imshow(window, frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                show_skeleton = not show_skeleton
            elif key == ord('c'):
                word_builder.clear()
                print("Text cleared!")
            elif key == 8:  # Backspace
                word_builder.backspace()
                print(f"Deleted last char | Text: '{word_builder.get_text()}'")
    
    except KeyboardInterrupt:
        pass
    
    finally:
        if detector:
            detector.close()
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        
        # Show final text
        final_text = word_builder.get_text()
        print("\n" + "=" * 60)
        print("📝 FINAL TEXT:")
        print(f"   '{final_text}'")
        print("=" * 60)
        print("✅ Application closed")


if __name__ == "__main__":
    main()

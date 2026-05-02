import os
import importlib
import sys
import warnings

# Set noisy runtime logs to the minimum before MediaPipe/TensorFlow import.
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning)

CAM_W, CAM_H = 640, 480
PROC_W, PROC_H = 320, 240
CAM_WIN_X = 20
CAM_WIN_Y = 20

# Multi-hand + preprocessing
MAX_NUM_HANDS = 2
PRIMARY_HAND = "right"  # right | left | largest
USE_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (6, 6)
ADAPTIVE_FRAME_SKIP = False

# Cursor
SMOOTHING = 0.32
CURSOR_SMOOTH_FRAMES = 3
CAM_MARGIN_X = 0.10
CAM_MARGIN_Y = 0.10
DEADZONE = 1.0

# Hand landmark smoothing (old*alpha + raw*(1-alpha)).
# Lower alpha = lower latency, higher alpha = more smoothing.
LANDMARK_SMOOTHING = 0.10
PREDICTION_MS = 35
PREDICTION_MAX_NORM_STEP = 0.045
VELOCITY_SMOOTHING = 0.55

# Threshold smoothing for hand-size normalization.
HAND_SCALE_SMOOTHING = 0.70

# Finger state detection
FINGER_STATE_METHOD = "angle"  # angle | tip
FINGER_ANGLE_THRESH = 165.0

# Gesture thresholds
PINCH_RATIO = 0.225
DRAW_PINCH_RATIO = 0.250
RELEASE_RATIO = 0.350
SCROLL_RATIO = 0.175
ZOOM_RATIO = 0.425

PINCH_THR_MIN, PINCH_THR_MAX = 0.02, 0.12
DRAW_PINCH_THR_MIN, DRAW_PINCH_THR_MAX = 0.03, 0.15
RELEASE_THR_MIN, RELEASE_THR_MAX = 0.03, 0.15
SCROLL_THR_MIN, SCROLL_THR_MAX = 0.02, 0.12
ZOOM_THR_MIN, ZOOM_THR_MAX = 0.08, 0.28

# Scroll
SCROLL_ENTRY_FRAMES = 3
SCROLL_COOLDOWN = 0.08
SCROLL_SENSITIVITY = 6

# Right click
RIGHT_CLICK_HOLD_SEC = 0.45

# Window switch (Mode 1: ring + thumb hold)
WINDOW_SWITCH_HOLD_SEC = 0.24
WINDOW_SWITCH_STEP_X = 0.035
WINDOW_SWITCH_STEP_COOLDOWN = 0.09

# Tab swipe
SWIPE_WINDOW_SEC = 0.35
SWIPE_MIN_X = 0.18
SWIPE_MAX_Y = 0.12
SWIPE_COOLDOWN = 1.2

# Zoom
ZOOM_COOLDOWN = 0.15

# Mode switch
MODE_SWITCH_FRAMES = 35

# Mode 3 viewer
MODE3_SEND_FPS = 60
MODE3_SCALE_MIN = 0.35
MODE3_SCALE_MAX = 3.0
MODE3_TRANSLATE_GAIN = 2.4
MODE3_ROTATE_GAIN = 4.2

# Display
DISPLAY_FPS = True
DISPLAY_VISIBILITY_POLL_SEC = 0.2
DISPLAY_EVERY_N = 1

# Optional gesture model (KNN)
USE_GESTURE_MODEL = False
GESTURE_MODEL_PATH = "gesture_model.npz"
MODEL_SCROLL_LABELS = {"SCROLL"}
MODEL_ZOOM_LABELS = {"ZOOM"}
MODEL_MODE_LABELS = {"MODE_SWITCH"}

# Screen editor
BRUSH_MIN = 3
BRUSH_MAX = 18
DRAW_SMOOTHING = 0.46
DRAW_MIN_POINT_DIST = 3
DRAW_INTERP_STEP_PX = 5
DRAW_OEF_MINCUTOFF = 1.15
DRAW_OEF_BETA = 0.018
ERASER_SIZE = 40
CLEAR_HOLD_SEC = 1.5
CLICK_DRAG_HOLD_SEC = 0.22

EDITOR_COLORS_RGB = [
    (255, 60, 60),
    (255, 160, 30),
    (255, 230, 30),
    (60, 220, 60),
    (60, 140, 255),
    (180, 60, 220),
    (255, 255, 255),
]
EDITOR_COLORS_BGR = [(b, g, r) for (r, g, b) in EDITOR_COLORS_RGB]
EDITOR_COLORS_HEX = [f"#{r:02x}{g:02x}{b:02x}" for (r, g, b) in EDITOR_COLORS_RGB]

TRANSPARENT_KEY = "#010101"


def reload_settings():
    mod = sys.modules[__name__]
    importlib.reload(mod)
    return mod

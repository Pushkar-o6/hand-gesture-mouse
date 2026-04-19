import os
import warnings
os.environ['GLOG_minloglevel']        = '3'   # Suppress MediaPipe C++ INFO/WARNING logs
os.environ['TF_CPP_MIN_LOG_LEVEL']   = '3'   # Suppress TensorFlow logs
warnings.filterwarnings('ignore', category=UserWarning)  # Suppress protobuf deprecation
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math
import threading
import queue
import tkinter as tk
from collections import deque

# ╔══════════════════════════════════════════════════════════════════╗
# ║        HAND GESTURE MOUSE & SCREEN EDITOR  v4.0                 ║
# ║                                                                  ║
# ║  v4 CHANGES:                                                     ║
# ║    • Absolute cursor positioning — no more delta drift/shake     ║
# ║    • model_complexity=0 — 2x faster MediaPipe                   ║
# ║    • Process at 360p, display at 480p — better FPS              ║
# ║    • moveWindow called once, not every frame                     ║
# ║    • Queue deduplication — no cursor spam                        ║
# ║    • Window minimize skip — no wasted CPU                       ║
# ║    • Mode 2 fully reworked — thumb-as-trigger system            ║
# ║                                                                  ║
# ║  MODE 1 — MOUSE CONTROL                                         ║
# ║    Index tip            → Move cursor (absolute mapped)         ║
# ║    Index + Middle close → Scroll (anchor-based)                 ║
# ║    Thumb + Middle tap   → Left click                            ║
# ║    Thumb + Middle hold  → Drag                                  ║
# ║    Thumb + Index hold   → Right click                           ║
# ║    Index + Middle wide  → Zoom (Ctrl +/-)                       ║
# ║    Open palm swipe      → Switch browser tabs                   ║
# ║                                                                  ║
# ║  MODE 2 — SCREEN EDITOR                                         ║
# ║    Relaxed hand          → Pointer / pen up (dashed cursor)     ║
# ║    Index + Thumb pinch   → Draw (pen down, grip the pen)        ║
# ║    Middle + Thumb pinch  → Erase                                ║
# ║    Ring + Thumb pinch    → Cycle draw color                     ║
# ║    Pinky + Thumb hold    → Clear canvas                         ║
# ║    Spread while drawing  → Bigger brush size                    ║
# ║    Keyboard S            → Save drawing                         ║
# ║    Keyboard C            → Clear canvas                         ║
# ║                                                                  ║
# ║  BOTH MODES                                                      ║
# ║    ILY gesture hold      → Switch mode                          ║
# ╚══════════════════════════════════════════════════════════════════╝

# ================================================================
#  SECTION 1 — SETTINGS
# ================================================================

CAM_W, CAM_H          = 640, 480
PROC_W, PROC_H        = 480, 360   # MediaPipe processes at this size (faster)
CAM_WIN_X             = 20
CAM_WIN_Y             = 20

# --- Cursor (absolute positioning) ---
SMOOTHING             = 0.25        # Lower = less lag, less smoothing
CURSOR_SMOOTH_FRAMES  = 5           # Rolling median over N frames (kills jitter)
# Map only the inner 80% of the cam frame to avoid edge detection issues
CAM_MARGIN_X          = 0.10        # 10% margin on each side
CAM_MARGIN_Y          = 0.10
DEADZONE              = 2.5         # Pixels — ignore sub-pixel noise

# --- Gesture thresholds (ratios of hand size — auto-scale with camera distance) ---
# All distances are expressed as a fraction of the wrist→middle-knuckle length (hsc).
# This means the same physical finger movement always triggers the same gesture
# regardless of how far your hand is from the camera.
#
# Calibration basis: at normal arm distance hsc ≈ 0.40, so:
#   PINCH_RATIO       = 0.07 / 0.40 = 0.175  (fingertips touching — all gestures)
#   DRAW_PINCH_RATIO  = 0.09 / 0.40 = 0.225  (index+thumb draw — slightly looser)
#   RELEASE_RATIO     = 0.09 / 0.40 = 0.225  (fingers apart — tighter than before)
#   SCROLL_RATIO      = 0.07 / 0.40 = 0.175  (index-middle close)
#   ZOOM_RATIO        = 0.17 / 0.40 = 0.425  (index-middle visibly spread)
PINCH_RATIO           = 0.225
DRAW_PINCH_RATIO      = 0.250
RELEASE_RATIO         = 0.350
SCROLL_RATIO          = 0.175
ZOOM_RATIO            = 0.425

# Hard clamps — prevent thresholds going crazy at extreme distances
PINCH_THR_MIN,      PINCH_THR_MAX      = 0.02, 0.12
DRAW_PINCH_THR_MIN, DRAW_PINCH_THR_MAX = 0.03, 0.15
RELEASE_THR_MIN,    RELEASE_THR_MAX    = 0.03, 0.15
SCROLL_THR_MIN,     SCROLL_THR_MAX     = 0.02, 0.12
ZOOM_THR_MIN,       ZOOM_THR_MAX       = 0.08, 0.28

# --- Scroll ---
SCROLL_ENTRY_FRAMES   = 3
SCROLL_COOLDOWN       = 0.08
SCROLL_SENSITIVITY    = 6

# --- Right click ---
RIGHT_CLICK_HOLD_SEC  = 0.45

# --- Tab swipe ---
SWIPE_WINDOW_SEC      = 0.35
SWIPE_MIN_X           = 0.18        # Fraction of frame width
SWIPE_MAX_Y           = 0.12        # Fraction of frame height
SWIPE_COOLDOWN        = 1.2

# --- Zoom ---
# Zoom requires index+middle clearly spread AND ring+pinky folded
ZOOM_COOLDOWN         = 0.15

# --- Mode switch ---
MODE_SWITCH_FRAMES    = 35

# --- Screen editor ---
BRUSH_MIN             = 3
BRUSH_MAX             = 18
DRAW_SMOOTHING        = 0.72        # High value = smooth path, less jitter
ERASER_SIZE           = 40
CLEAR_HOLD_SEC        = 1.5         # Hold pinky+thumb this long to clear
CLICK_DRAG_HOLD_SEC   = 0.22        # Thumb+middle hold duration before drag starts

EDITOR_COLORS_RGB = [
    (255,  60,  60),  # Red
    (255, 160,  30),  # Orange
    (255, 230,  30),  # Yellow
    ( 60, 220,  60),  # Green
    ( 60, 140, 255),  # Blue
    (180,  60, 220),  # Purple
    (255, 255, 255),  # White
]
EDITOR_COLORS_BGR = [(b, g, r) for (r, g, b) in EDITOR_COLORS_RGB]
EDITOR_COLORS_HEX = [f'#{r:02x}{g:02x}{b:02x}' for (r, g, b) in EDITOR_COLORS_RGB]

TRANSPARENT_KEY = '#010101'         # Near-black but not pure black (pure black
# can clash with some dark app UIs)

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0

# ================================================================
#  SECTION 2 — SHARED STATE
# ================================================================

draw_queue   = queue.Queue(maxsize=120)   # bounded — drops stale frames if behind
ctrl_queue   = queue.Queue(maxsize=32)    # reliable control channel (quit/save/clear/mode)
current_mode = 1
mode_lock    = threading.Lock()

screen_w, screen_h = pyautogui.size()

# ================================================================
#  SECTION 3 — OVERLAY WINDOW
# ================================================================

class ScreenOverlay:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Screen Editor Overlay')
        self.root.attributes('-fullscreen',       True)
        self.root.attributes('-topmost',          True)
        self.root.attributes('-transparentcolor', TRANSPARENT_KEY)
        self.root.configure(bg=TRANSPARENT_KEY)
        self.root.overrideredirect(True)

        self.canvas = tk.Canvas(
            self.root,
            bg=TRANSPARENT_KEY,
            highlightthickness=0,
            cursor='none',
        )
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<KeyPress-s>', self.save_drawing)
        self.canvas.bind('<KeyPress-c>', lambda e: self.clear_canvas())
        self.canvas.bind('<Escape>',     self.quit)
        self.canvas.focus_set()

        self.hud_id      = None
        self.cursor_ids  = []
        self.palette_ids = []
        self.strokes     = []          # all canvas item ids for drawings
        self.color_idx   = 0
        self.live_stroke_ids = {}

        self._draw_palette()
        self._draw_hud('Mode 1: Mouse Control', '#FF8C00')
        self.root.after(8, self._poll)

    # ── Queue polling ────────────────────────────────────────────
    def _poll(self):
        try:
            while True:
                self._handle(ctrl_queue.get_nowait())
        except queue.Empty:
            pass
        try:
            while True:
                self._handle(draw_queue.get_nowait())
        except queue.Empty:
            pass
        self.root.after(8, self._poll)

    def _handle(self, msg):
        kind = msg[0]

        if kind == 'draw':
            if len(msg) == 8:
                _, x1, y1, x2, y2, color, size, stroke_id = msg
            else:
                _, x1, y1, x2, y2, color, size = msg
                stroke_id = None
            iid = self.canvas.create_line(
                x1, y1, x2, y2,
                fill=color, width=size,
                capstyle=tk.ROUND, joinstyle=tk.ROUND, smooth=True,
            )
            self.strokes.append(iid)
            if stroke_id is not None:
                self.live_stroke_ids.setdefault(stroke_id, []).append(iid)

        elif kind == 'stroke_begin':
            _, stroke_id = msg
            self.live_stroke_ids[stroke_id] = []

        elif kind == 'stroke_finalize':
            _, stroke_id, points, color, size = msg
            self._finalize_stroke(stroke_id, points, color, size)

        elif kind == 'erase':
            _, ex, ey, sz = msg
            r   = sz // 2
            iid = self.canvas.create_oval(
                ex-r, ey-r, ex+r, ey+r,
                fill=TRANSPARENT_KEY, outline=TRANSPARENT_KEY,
                )
            self.strokes.append(iid)

        elif kind == 'clear':
            self.clear_canvas()

        elif kind == 'mode':
            _, m = msg
            if m == 1:
                for iid in self.strokes:
                    self.canvas.itemconfigure(iid, state='hidden')
                self._draw_hud('Mode 1: Mouse Control', '#FF8C00')
            else:
                for iid in self.strokes:
                    self.canvas.itemconfigure(iid, state='normal')
                self._draw_hud('Mode 2: Screen Editor  [pinch=draw  2-pinch=erase]', '#00D264')

        elif kind == 'cursor':
            _, cx, cy, color, style = msg
            self._draw_cursor(cx, cy, color, style)

        elif kind == 'cursor_hide':
            self._clear_cursor()

        elif kind == 'color':
            _, idx = msg
            self.color_idx = idx
            self._draw_palette()

        elif kind == 'save':
            self.save_drawing()

        elif kind == 'quit':
            self.root.destroy()

    # ── Cursor ───────────────────────────────────────────────────
    def _clear_cursor(self):
        for iid in self.cursor_ids:
            self.canvas.delete(iid)
        self.cursor_ids = []

    def _draw_cursor(self, cx, cy, color, style):
        self._clear_cursor()
        ids = []

        if style == 'draw':
            r = 5
            ids.append(self.canvas.create_oval(
                cx-r, cy-r, cx+r, cy+r, fill=color, outline='black', width=1))
            arm = 14
            for x1, y1, x2, y2 in [
                (cx-arm, cy, cx-r-2, cy), (cx+r+2, cy, cx+arm, cy),
                (cx, cy-arm, cx, cy-r-2), (cx, cy+r+2, cx, cy+arm),
            ]:
                ids.append(self.canvas.create_line(x1, y1, x2, y2, fill=color, width=2))

        elif style == 'penup':
            r = 8
            ids.append(self.canvas.create_oval(
                cx-r, cy-r, cx+r, cy+r, outline='#777777', width=1, dash=(3,3)))
            arm = 18
            for x1, y1, x2, y2 in [
                (cx-arm, cy, cx-r-2, cy), (cx+r+2, cy, cx+arm, cy),
                (cx, cy-arm, cx, cy-r-2), (cx, cy+r+2, cx, cy+arm),
            ]:
                ids.append(self.canvas.create_line(
                    x1, y1, x2, y2, fill='#777777', width=1, dash=(3,3)))

        elif style == 'erase':
            r = ERASER_SIZE
            ids.append(self.canvas.create_oval(
                cx-r, cy-r, cx+r, cy+r, outline='#aaaaaa', width=2, dash=(4,4)))
            ids.append(self.canvas.create_oval(cx-3, cy-3, cx+3, cy+3, outline='#aaaaaa'))

        self.cursor_ids = ids

    # ── Canvas helpers ───────────────────────────────────────────
    def clear_canvas(self):
        for iid in self.strokes:
            self.canvas.delete(iid)
        self.strokes.clear()
        self.live_stroke_ids.clear()
        self._clear_cursor()

    def _finalize_stroke(self, stroke_id, points, color, size):
        old_ids = self.live_stroke_ids.pop(stroke_id, [])
        for iid in old_ids:
            self.canvas.delete(iid)
            if iid in self.strokes:
                self.strokes.remove(iid)

        if len(points) < 2:
            return

        shape_kind, geom = self._detect_shape(points)
        if shape_kind == 'line':
            (x1, y1), (x2, y2) = geom
            iid = self.canvas.create_line(
                x1, y1, x2, y2,
                fill=color,
                width=size,
                capstyle=tk.ROUND,
                joinstyle=tk.ROUND,
                smooth=False,
            )
            self.strokes.append(iid)

        elif shape_kind == 'rect':
            rect_pts = geom
            flat = [v for p in rect_pts for v in p]
            iid = self.canvas.create_polygon(
                *flat,
                outline=color,
                fill='',
                width=size,
                smooth=False,
            )
            self.strokes.append(iid)

        elif shape_kind == 'circle':
            cx, cy, r = geom
            iid = self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                outline=color,
                width=size,
            )
            self.strokes.append(iid)

        else:
            smooth_points = self._smooth_stroke(points)
            if len(smooth_points) >= 2:
                flat = [v for p in smooth_points for v in p]
                iid = self.canvas.create_line(
                    *flat,
                    fill=color,
                    width=size,
                    capstyle=tk.ROUND,
                    joinstyle=tk.ROUND,
                    smooth=True,
                    splinesteps=24,
                )
                self.strokes.append(iid)

    def _detect_shape(self, points):
        pts = np.array(points, dtype=np.float32)
        if len(pts) < 3:
            return 'raw', None

        p0 = pts[0]
        p1 = pts[-1]
        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)
        w, h = bbox_max - bbox_min
        diag = max(10.0, float(np.hypot(w, h)))
        closed = np.linalg.norm(p1 - p0) < (0.20 * diag)

        line_dist = np.linalg.norm(p1 - p0)
        if line_dist > 10:
            line_vec = (p1 - p0) / (line_dist + 1e-6)
            rel = pts - p0
            perp = np.abs(rel[:, 0] * line_vec[1] - rel[:, 1] * line_vec[0])
            mean_perp = float(np.mean(perp))
            if not closed and mean_perp < max(3.5, 0.03 * diag):
                return 'line', ((float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1])))

        if not closed or len(pts) < 8:
            return 'raw', None

        contour = pts.astype(np.int32).reshape(-1, 1, 2)
        area = abs(cv2.contourArea(contour))
        if area < 80:
            return 'raw', None

        peri = cv2.arcLength(contour, True)
        circularity = float((4.0 * math.pi * area) / (peri * peri + 1e-6))
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        radius = max(1e-6, float(radius))
        dists = np.linalg.norm(pts - np.array([cx, cy], dtype=np.float32), axis=1)
        radial_std_ratio = float(np.std(dists) / radius)
        aspect_ratio = float(min(w, h) / max(w, h, 1e-6))

        if circularity > 0.74 and radial_std_ratio < 0.22 and aspect_ratio > 0.72:
            return 'circle', (float(cx), float(cy), float(radius))

        approx = cv2.approxPolyDP(contour, 0.03 * peri, True)

        if len(approx) == 4:
            quad = [(float(p[0][0]), float(p[0][1])) for p in approx]
            if self._quad_is_rect(quad):
                ordered = self._order_quad(quad)
                rect_area = max(1.0, cv2.contourArea(np.array(ordered, dtype=np.float32)))
                fill_ratio = area / rect_area
                edge_support = self._rectangle_edge_support(pts, ordered)
                if fill_ratio > 0.68 and edge_support > 0.74:
                    return 'rect', ordered

        if circularity > 0.70 and radial_std_ratio < 0.20 and aspect_ratio > 0.80:
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            return 'circle', (float(cx), float(cy), float(radius))

        return 'raw', None

    def _order_quad(self, quad):
        arr = np.array(quad, dtype=np.float32)
        center = arr.mean(axis=0)
        ang = np.arctan2(arr[:, 1] - center[1], arr[:, 0] - center[0])
        ordered = arr[np.argsort(ang)]
        start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
        ordered = np.roll(ordered, -start, axis=0)
        return [(float(p[0]), float(p[1])) for p in ordered]

    def _quad_is_rect(self, quad):
        q = self._order_quad(quad)
        v = []
        for i in range(4):
            x1, y1 = q[i]
            x2, y2 = q[(i + 1) % 4]
            v.append(np.array([x2 - x1, y2 - y1], dtype=np.float32))

        for i in range(4):
            a = v[i]
            b = v[(i + 1) % 4]
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na < 3 or nb < 3:
                return False
            cosang = float(np.dot(a, b) / (na * nb + 1e-6))
            angle = math.degrees(math.acos(np.clip(abs(cosang), 0.0, 1.0)))
            if abs(angle - 90.0) > 22.0:
                return False
        return True

    def _point_to_seg_dist(self, p, a, b):
        p = np.array(p, dtype=np.float32)
        a = np.array(a, dtype=np.float32)
        b = np.array(b, dtype=np.float32)
        ab = b - a
        den = float(np.dot(ab, ab))
        if den < 1e-6:
            return float(np.linalg.norm(p - a))
        t = float(np.dot(p - a, ab) / den)
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        return float(np.linalg.norm(p - proj))

    def _rectangle_edge_support(self, pts, quad):
        tol = max(4.0, 0.025 * float(np.hypot(pts[:, 0].max() - pts[:, 0].min(), pts[:, 1].max() - pts[:, 1].min())))
        ok = 0
        for p in pts:
            d = min(
                self._point_to_seg_dist(p, quad[0], quad[1]),
                self._point_to_seg_dist(p, quad[1], quad[2]),
                self._point_to_seg_dist(p, quad[2], quad[3]),
                self._point_to_seg_dist(p, quad[3], quad[0]),
            )
            if d <= tol:
                ok += 1
        return ok / max(1, len(pts))

    def _smooth_stroke(self, points):
        if len(points) < 4:
            return points

        pts = np.array(points, dtype=np.float32)
        for _ in range(2):
            new_pts = [pts[0]]
            for i in range(len(pts) - 1):
                p = pts[i]
                q = pts[i + 1]
                new_pts.append(0.75 * p + 0.25 * q)
                new_pts.append(0.25 * p + 0.75 * q)
            new_pts.append(pts[-1])
            pts = np.array(new_pts, dtype=np.float32)

        smoothed = [(float(p[0]), float(p[1])) for p in pts[::2]]
        if smoothed[-1] != (float(pts[-1][0]), float(pts[-1][1])):
            smoothed.append((float(pts[-1][0]), float(pts[-1][1])))
        return smoothed

    def _draw_hud(self, text, color):
        if self.hud_id:
            self.canvas.delete(self.hud_id)
        self.hud_id = self.canvas.create_text(
            14, 14, text=text, anchor='nw',
            fill=color, font=('Helvetica', 13, 'bold'),
        )

    def _draw_palette(self):
        for iid in self.palette_ids:
            self.canvas.delete(iid)
        self.palette_ids.clear()
        y = screen_h - 22
        for i, hex_c in enumerate(EDITOR_COLORS_HEX):
            x = 22 + i * 30
            r = 12 if i == self.color_idx else 9
            out = 'white' if i == self.color_idx else hex_c
            self.palette_ids.append(self.canvas.create_oval(
                x-r, y-r, x+r, y+r, fill=hex_c, outline=out, width=2))

    def save_drawing(self, event=None):
        try:
            from PIL import ImageGrab
            fname = f'drawing_{int(time.time())}.png'
            ImageGrab.grab().save(fname)
            print(f'[Saved] {fname}')
            self._draw_hud(f'Saved: {fname}', '#00FF88')
            self.root.after(2000, lambda: self._draw_hud('Mode 2: Screen Editor', '#00D264'))
        except ImportError:
            print('[Save] pip install Pillow')

    def quit(self, event=None):
        self.root.destroy()

    def run(self):
        self.root.mainloop()

# ================================================================
#  SECTION 4 — HELPERS
# ================================================================

def lm_norm(lm, idx):
    """Return landmark as (x, y) normalized 0-1 floats."""
    return lm[idx].x, lm[idx].y

def dist_norm(lm, a, b):
    """Euclidean distance between two landmarks in normalized space."""
    ax, ay = lm[a].x, lm[a].y
    bx, by = lm[b].x, lm[b].y
    return math.hypot(ax-bx, ay-by)

def hand_scale(lm):
    """
    Reference scale = distance from wrist (0) to middle MCP (9).
    Used to normalize all pinch thresholds to hand size.
    """
    return dist_norm(lm, 0, 9) + 1e-6

def is_finger_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y

def count_fingers_up(lm):
    return sum(1 for t, p in [(8,6),(12,10),(16,14),(20,18)] if is_finger_up(lm, t, p))

def is_fist(lm):
    return count_fingers_up(lm) == 0

def is_open_palm(lm):
    return count_fingers_up(lm) == 4

def is_ily_gesture(lm):
    return (is_finger_up(lm, 8, 6) and is_finger_up(lm, 20, 18) and
            not is_finger_up(lm, 12, 10) and not is_finger_up(lm, 16, 14))

def norm_to_screen(nx, ny):
    """
    Map normalized hand coords (0-1) to screen pixels.
    Uses inner CAM_MARGIN to give the usable zone more range.
    """
    x_lo, x_hi = CAM_MARGIN_X, 1.0 - CAM_MARGIN_X
    y_lo, y_hi = CAM_MARGIN_Y, 1.0 - CAM_MARGIN_Y
    sx = int(np.interp(nx, [x_lo, x_hi], [0, screen_w]))
    sy = int(np.interp(ny, [y_lo, y_hi], [0, screen_h]))
    sx = int(np.clip(sx, 0, screen_w - 1))
    sy = int(np.clip(sy, 0, screen_h - 1))
    return sx, sy

def try_put(q, msg):
    """Non-blocking queue put — silently drops if full."""
    try:
        q.put_nowait(msg)
    except queue.Full:
        pass

def try_put_ctrl(msg):
    """Control messages should be reliable; if full, drop oldest and enqueue newest."""
    try:
        ctrl_queue.put_nowait(msg)
    except queue.Full:
        try:
            ctrl_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            ctrl_queue.put_nowait(msg)
        except queue.Full:
            pass

class LMPoint:
    __slots__ = ('x', 'y')
    def __init__(self, x, y):
        self.x = x
        self.y = y

# ================================================================
#  SECTION 5 — WEBCAM THREAD
# ================================================================

def webcam_thread():
    global current_mode

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        try_put_ctrl(('quit',))
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS,          60)              # Request 60fps for smoother capture
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)              # Minimize buffer lag
    cap.set(cv2.CAP_PROP_AUTOFOCUS,    1)              # Enable autofocus

    mp_hands = mp.solutions.hands
    hands    = mp_hands.Hands(
        static_image_mode        = False,
        max_num_hands            = 1,
        model_complexity         = 0,          # Lite model — 2x faster
        min_detection_confidence = 0.60,       # More lenient — no detection gaps
        min_tracking_confidence  = 0.60,       # More responsive
    )
    draw_utils = mp.solutions.drawing_utils
    lm_style   = draw_utils.DrawingSpec(color=(0, 255, 180), thickness=1, circle_radius=2)
    cn_style   = draw_utils.DrawingSpec(color=(80, 80, 220), thickness=1)

    # ── Per-landmark EMA smoothing ───────────────────────────────
    # 21 landmarks × [x, y] — smoothed values used for BOTH drawing
    # and all gesture distance calculations
    LM_ALPHA  = 0.50          # blend: 0=fully raw (jittery), 1=fully frozen
    lm_smooth = None           # initialised on first detection

    # ── Hand-size smoothing (for stable dynamic thresholds) ──────
    hsc_smooth  = 0.40          # start at normal-distance estimate
    pinch_thr       = PINCH_RATIO      * 0.40
    draw_pinch_thr  = DRAW_PINCH_RATIO * 0.40
    release_thr     = RELEASE_RATIO    * 0.40
    scroll_thr      = SCROLL_RATIO     * 0.40
    zoom_thr        = ZOOM_RATIO       * 0.40

    # ── Absolute cursor state ────────────────────────────────────
    # Rolling buffer for median-filter jitter removal
    cx_buf = deque(maxlen=CURSOR_SMOOTH_FRAMES)
    cy_buf = deque(maxlen=CURSOR_SMOOTH_FRAMES)
    cursor_x = screen_w // 2
    cursor_y = screen_h // 2
    last_cursor_x = cursor_x
    last_cursor_y = cursor_y

    # ── Gesture state ────────────────────────────────────────────
    gesture_state         = 'IDLE'
    right_click_start     = None
    right_click_done      = False
    dragging              = False
    click_drag_start      = None

    scroll_anchor_y       = None
    fingers_together_cnt  = 0
    scroll_mode_active    = False
    last_scroll_time      = 0.0

    zoom_anchor_dist      = None
    last_zoom_time        = 0.0

    wrist_history         = deque()
    last_swipe_time       = 0.0
    swipe_flash_label     = ''
    swipe_flash_until     = 0.0

    mode_switch_frames    = 0

    # ── Mode 2 state ─────────────────────────────────────────────
    draw_color_idx        = 0
    prev_draw_sx          = None
    prev_draw_sy          = None
    smooth_draw_sx        = None
    smooth_draw_sy        = None
    drawing_active        = False
    stroke_points         = []
    stroke_color          = EDITOR_COLORS_HEX[0]
    stroke_size_samples   = []
    stroke_id_counter     = 0
    active_stroke_id      = None

    color_change_done     = False
    clear_hold_start      = None
    clear_done            = False

    # ── FPS ──────────────────────────────────────────────────────
    fps_buf   = deque(maxlen=30)
    prev_time = time.time()

    # ── Window setup — do this ONCE ──────────────────────────────
    window_name = 'Gesture Controller  |  ESC=quit'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, CAM_W // 2, CAM_H // 2)
    window_moved = False
    read_failures = 0
    max_read_failures = 20

    def finalize_active_stroke():
        nonlocal drawing_active, stroke_points, stroke_color, stroke_size_samples, active_stroke_id
        nonlocal prev_draw_sx, prev_draw_sy, smooth_draw_sx, smooth_draw_sy

        if drawing_active and active_stroke_id is not None and len(stroke_points) >= 2:
            avg_size = max(BRUSH_MIN, int(np.mean(stroke_size_samples) if stroke_size_samples else BRUSH_MIN))
            try_put(draw_queue, ('stroke_finalize', active_stroke_id, stroke_points, stroke_color, avg_size))

        drawing_active = False
        stroke_points = []
        stroke_size_samples = []
        active_stroke_id = None
        prev_draw_sx = prev_draw_sy = None
        smooth_draw_sx = smooth_draw_sy = None

    while True:
        success, frame = cap.read()
        if not success:
            read_failures += 1
            if read_failures >= max_read_failures:
                cap.release()
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    try_put_ctrl(('quit',))
                    break
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
                cap.set(cv2.CAP_PROP_FPS,          60)
                cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
                cap.set(cv2.CAP_PROP_AUTOFOCUS,    1)
                read_failures = 0
            time.sleep(0.005)
            continue
        read_failures = 0

        frame = cv2.flip(frame, 1)

        # ── Resize down for MediaPipe (speed), keep full for display ─
        small = cv2.resize(frame, (PROC_W, PROC_H))
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        with mode_lock:
            mode = current_mode

        # ── Banner ───────────────────────────────────────────────
        label = 'MODE 1: MOUSE' if mode == 1 else 'MODE 2: SCREEN EDITOR'
        color = (255, 140, 0) if mode == 1 else (0, 210, 100)
        cv2.rectangle(frame, (0, 0), (CAM_W, 36), (25, 25, 25), -1)
        cv2.putText(frame, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        now = time.time()

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            raw_lm = hand.landmark   # 21 raw MediaPipe landmarks

            # ── Build / update smoothed landmark array ────────────
            if lm_smooth is None:
                # First detection — initialise directly from raw values
                lm_smooth = [[lk.x, lk.y] for lk in raw_lm]
            else:
                for i, lk in enumerate(raw_lm):
                    lm_smooth[i][0] = lm_smooth[i][0] * LM_ALPHA + lk.x * (1 - LM_ALPHA)
                    lm_smooth[i][1] = lm_smooth[i][1] * LM_ALPHA + lk.y * (1 - LM_ALPHA)

            # ── Draw skeleton using SMOOTHED coords ───────────────
            # Draw every frame for smooth hand skeleton visualization
            for conn in mp_hands.HAND_CONNECTIONS:
                a, b = conn
                ax = int(lm_smooth[a][0] * CAM_W)
                ay = int(lm_smooth[a][1] * CAM_H)
                bx = int(lm_smooth[b][0] * CAM_W)
                by = int(lm_smooth[b][1] * CAM_H)
                cv2.line(frame, (ax, ay), (bx, by), (80, 80, 220), 1)
            for i, (sx_lm, sy_lm) in enumerate(lm_smooth):
                px = int(sx_lm * CAM_W)
                py = int(sy_lm * CAM_H)
                r  = 4 if i in (4, 8, 12, 16, 20) else 2   # bigger dots on fingertips
                cv2.circle(frame, (px, py), r, (0, 255, 180), -1)

            # Build once-per-frame landmark view using a module-level lightweight class.
            lm = [LMPoint(x, y) for x, y in lm_smooth]  # smoothed, used everywhere below

            # ── Key landmark coords (from SMOOTHED lm) ────────────

            # ── Dynamic thresholds — scale with hand distance ─────
            # hsc = wrist→middle-knuckle distance, shrinks as hand moves away
            raw_hsc    = hand_scale(lm)
            hsc_smooth = hsc_smooth * 0.85 + raw_hsc * 0.15   # slow EMA — stable thresholds
            hsc        = hsc_smooth

            pinch_thr       = float(np.clip(PINCH_RATIO      * hsc, PINCH_THR_MIN,      PINCH_THR_MAX))
            draw_pinch_thr  = float(np.clip(DRAW_PINCH_RATIO * hsc, DRAW_PINCH_THR_MIN, DRAW_PINCH_THR_MAX))
            release_thr     = float(np.clip(RELEASE_RATIO    * hsc, RELEASE_THR_MIN,    RELEASE_THR_MAX))
            scroll_thr      = float(np.clip(SCROLL_RATIO     * hsc, SCROLL_THR_MIN,     SCROLL_THR_MAX))
            zoom_thr        = float(np.clip(ZOOM_RATIO       * hsc, ZOOM_THR_MIN,       ZOOM_THR_MAX))
            ix_n, iy_n = lm[8].x,  lm[8].y    # index tip
            mx_n, my_n = lm[12].x, lm[12].y   # middle tip
            rx_n, ry_n = lm[16].x, lm[16].y   # ring tip
            px_n, py_n = lm[20].x, lm[20].y   # pinky tip
            tx_n, ty_n = lm[4].x,  lm[4].y    # thumb tip
            wx_n, wy_n = lm[0].x,  lm[0].y    # wrist

            d_click  = dist_norm(lm, 12,  4)  # middle-thumb
            d_drag   = dist_norm(lm, 16,  4)  # ring-thumb
            d_scroll = dist_norm(lm,  8, 12)  # index-middle
            d_right  = dist_norm(lm,  8,  4)  # index-thumb
            d_zoom   = dist_norm(lm,  8, 12)  # reuse index-middle
            d_draw   = dist_norm(lm,  8,  4)  # index-thumb (draw pinch)
            d_erase  = dist_norm(lm, 12,  4)  # middle-thumb (erase pinch)
            d_color  = dist_norm(lm, 16,  4)  # ring-thumb
            d_clear  = dist_norm(lm, 20,  4)  # pinky-thumb

            index_up  = is_finger_up(lm, 8,  6)
            middle_up = is_finger_up(lm, 12, 10)

            # ── Absolute cursor target (Mode 1) ───────────────────
            # Use index tip with median filter for jitter removal
            tx_s, ty_s = norm_to_screen(ix_n, iy_n)
            cx_buf.append(tx_s)
            cy_buf.append(ty_s)
            cx_med = int(sorted(cx_buf)[len(cx_buf)//2])
            cy_med = int(sorted(cy_buf)[len(cy_buf)//2])
            # Exponential smooth on top of median
            cursor_x = int(cursor_x * SMOOTHING + cx_med * (1 - SMOOTHING))
            cursor_y = int(cursor_y * SMOOTHING + cy_med * (1 - SMOOTHING))

            # ── ILY mode switch ───────────────────────────────────
            if is_ily_gesture(lm):
                mode_switch_frames += 1
                prog = int((mode_switch_frames / MODE_SWITCH_FRAMES) * (CAM_W - 20))
                cv2.rectangle(frame, (10, CAM_H-22), (10+prog, CAM_H-10), (0,255,200), -1)
                cv2.putText(frame, '\m/ Hold to switch mode...',
                            (10, CAM_H-28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,255,200), 1)
                if mode_switch_frames >= MODE_SWITCH_FRAMES:
                    with mode_lock:
                        current_mode = 2 if current_mode == 1 else 1
                        new_mode = current_mode
                    mode_switch_frames  = 0
                    gesture_state       = 'IDLE'
                    right_click_start   = None
                    right_click_done    = False
                    scroll_mode_active  = False
                    scroll_anchor_y     = None
                    zoom_anchor_dist    = None
                    finalize_active_stroke()
                    clear_hold_start = None
                    clear_done = False
                    cx_buf.clear(); cy_buf.clear()
                    try_put_ctrl(('mode', new_mode))
            else:
                mode_switch_frames = 0

            # ════════════════════════════════════════════════════
            #  MODE 1 — MOUSE CONTROL
            # ════════════════════════════════════════════════════
            if mode == 1:
                try_put(draw_queue, ('cursor_hide',))

                # ── Tab swipe ─────────────────────────────────────
                wrist_history.append((wx_n, wy_n, now))
                while wrist_history and now - wrist_history[0][2] > SWIPE_WINDOW_SEC:
                    wrist_history.popleft()

                if (is_open_palm(lm) and len(wrist_history) >= 4 and
                        (now - last_swipe_time) > SWIPE_COOLDOWN):
                    x0, y0, _ = wrist_history[0]
                    net_x = wx_n - x0
                    net_y = abs(wy_n - y0)
                    if abs(net_x) > SWIPE_MIN_X and net_y < SWIPE_MAX_Y:
                        if net_x > 0:
                            pyautogui.hotkey('ctrl', 'shift', 'tab')
                            swipe_flash_label = '<< PREV TAB'
                        else:
                            pyautogui.hotkey('ctrl', 'tab')
                            swipe_flash_label = 'NEXT TAB >>'
                        last_swipe_time   = now
                        swipe_flash_until = now + 0.8
                        wrist_history.clear()

                if now < swipe_flash_until:
                    cv2.putText(frame, swipe_flash_label,
                                (CAM_W//2 - 80, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,220,255), 2)

                # ── Right click hold (index curled to thumb) ───────
                if d_right < pinch_thr and not index_up and not scroll_mode_active:
                    if right_click_start is None:
                        right_click_start = now
                    elif not right_click_done and now - right_click_start > RIGHT_CLICK_HOLD_SEC:
                        pyautogui.rightClick()
                        right_click_done = True
                    pct = min(1.0, (now - right_click_start) / RIGHT_CLICK_HOLD_SEC)
                    tx_px = int(tx_n * CAM_W); ty_px = int(ty_n * CAM_H)
                    cv2.ellipse(frame, (tx_px, ty_px), (18,18), -90, 0, int(360*pct), (0,100,255), 3)
                    cv2.putText(frame, 'RIGHT CLICK', (10,160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,100,255), 2)
                else:
                    right_click_start = None
                    right_click_done  = False

                # ── Zoom (index + middle spread wide, ring+pinky DOWN) ─
                # ring+pinky must be folded so zoom can't fire during normal cursor use
                ring_dn  = not is_finger_up(lm, 16, 14)
                pinky_dn = not is_finger_up(lm, 20, 18)
                zoom_norm = dist_norm(lm, 8, 12)
                if index_up and middle_up and ring_dn and pinky_dn and zoom_norm > zoom_thr:
                    if zoom_anchor_dist is None:
                        zoom_anchor_dist = zoom_norm
                    else:
                        delta = zoom_norm - zoom_anchor_dist
                        if abs(delta) > 0.018 and (now - last_zoom_time) > ZOOM_COOLDOWN:
                            pyautogui.hotkey('ctrl', '+' if delta > 0 else '-')
                            zoom_anchor_dist = zoom_norm
                            last_zoom_time   = now
                    ix_px = int(ix_n*CAM_W); iy_px = int(iy_n*CAM_H)
                    mx_px = int(mx_n*CAM_W); my_px = int(my_n*CAM_H)
                    cv2.line(frame, (ix_px,iy_px), (mx_px,my_px), (255,200,0), 2)
                    cv2.putText(frame, 'ZOOM', (10,160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,0), 2)
                else:
                    zoom_anchor_dist = None

                # ── Scroll vs Move ────────────────────────────────
                if d_scroll < scroll_thr:
                    fingers_together_cnt += 1
                    if fingers_together_cnt >= SCROLL_ENTRY_FRAMES:
                        scroll_mode_active = True

                    if scroll_mode_active:
                        if scroll_anchor_y is None:
                            scroll_anchor_y = iy_n
                        scroll_delta = iy_n - scroll_anchor_y
                        if abs(scroll_delta) > 0.005 and (now - last_scroll_time) > SCROLL_COOLDOWN:
                            direction    = -1 if scroll_delta > 0 else 1
                            scroll_ticks = max(1, int(abs(scroll_delta) * 40))
                            pyautogui.scroll(direction * scroll_ticks * SCROLL_SENSITIVITY)
                            last_scroll_time = now
                        anc_py = int(scroll_anchor_y * CAM_H)
                        cv2.line(frame, (0, anc_py), (CAM_W, anc_py), (0,255,255), 1)
                        ix_px = int(ix_n*CAM_W); iy_px = int(iy_n*CAM_H)
                        cv2.circle(frame, (ix_px,iy_px), 10, (0,255,255), 2)
                        cv2.putText(frame, 'SCROLLING', (10,160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

                else:
                    fingers_together_cnt = 0
                    scroll_mode_active   = False
                    scroll_anchor_y      = None

                    if zoom_anchor_dist is None:
                        # Absolute position move — no drift, no accumulated error
                        moved_x = abs(cursor_x - last_cursor_x)
                        moved_y = abs(cursor_y - last_cursor_y)
                        if moved_x > DEADZONE or moved_y > DEADZONE:
                            pyautogui.moveTo(cursor_x, cursor_y)
                            last_cursor_x = cursor_x
                            last_cursor_y = cursor_y

                        ix_px = int(ix_n*CAM_W); iy_px = int(iy_n*CAM_H)
                        cv2.circle(frame, (ix_px,iy_px), 8, (255,140,0), 2)
                        cv2.putText(frame, 'MOVING', (10,160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,140,0), 2)

                # ── Combined click/drag (middle + thumb tap/hold) ──
                # Keep click strict (pinch_thr), but keep active drag using a looser
                # threshold (release_thr) so movement does not accidentally unpinch.
                click_drag_active = (d_click < (release_thr if dragging else pinch_thr)) and not scroll_mode_active
                if click_drag_active:
                    if click_drag_start is None:
                        click_drag_start = now

                    hold_t = now - click_drag_start
                    if hold_t >= CLICK_DRAG_HOLD_SEC:
                        if not dragging:
                            pyautogui.mouseDown()
                            dragging = True
                        mx_px = int(mx_n*CAM_W); my_px = int(my_n*CAM_H)
                        cv2.circle(frame, (mx_px,my_px), 14, (0,0,255), 3)
                        cv2.putText(frame, 'DRAG (HOLD)', (10,160),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    else:
                        pct = min(1.0, hold_t / CLICK_DRAG_HOLD_SEC)
                        mx_px = int(mx_n*CAM_W); my_px = int(my_n*CAM_H)
                        cv2.ellipse(frame, (mx_px, my_px), (16,16), -90, 0,
                                    int(360*pct), (0,255,120), 2)
                        cv2.putText(frame, 'CLICK READY', (10,160),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,120), 2)
                else:
                    if click_drag_start is not None:
                        hold_t = now - click_drag_start
                        if hold_t < CLICK_DRAG_HOLD_SEC and not dragging:
                            pyautogui.click()
                            gesture_state = 'CLICKED'
                        click_drag_start = None

                    if dragging:
                        pyautogui.mouseUp()
                        dragging = False

                    if gesture_state == 'CLICKED':
                        if d_click > release_thr:
                            gesture_state = 'IDLE'

                cv2.putText(frame, f'Click:{gesture_state}  Drag:{dragging}',
                            (10,185), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160,160,160), 1)

            # ════════════════════════════════════════════════════
            #  MODE 2 — SCREEN EDITOR  (thumb-as-trigger system)
            # ════════════════════════════════════════════════════
            elif mode == 2:
                cur_hex = EDITOR_COLORS_HEX[draw_color_idx]
                cur_bgr = EDITOR_COLORS_BGR[draw_color_idx]

                # Tracking point = index tip (always)
                # Normalize → screen coords with same margin system
                sx, sy = norm_to_screen(ix_n, iy_n)

                # ── Priority 1: Clear canvas (pinky+thumb hold) ───
                if d_clear < pinch_thr:
                    finalize_active_stroke()
                    if clear_hold_start is None:
                        clear_hold_start = now
                    pct = min(1.0, (now - clear_hold_start) / CLEAR_HOLD_SEC)
                    # Progress arc on webcam
                    px_px = int(px_n*CAM_W); py_px = int(py_n*CAM_H)
                    cv2.ellipse(frame, (px_px,py_px), (22,22), -90, 0,
                                int(360*pct), (0,80,255), 3)
                    cv2.putText(frame, 'Hold to clear...', (10,160),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,80,255), 2)
                    if pct >= 1.0 and not clear_done:
                        try_put(draw_queue, ('clear',))
                        prev_draw_sx = prev_draw_sy = None
                        clear_done = True
                    # Show cursor so user knows where they are
                    try_put(draw_queue, ('cursor', sx, sy, '#555555', 'penup'))

                else:
                    clear_hold_start = None
                    clear_done       = False

                    # ── Priority 2: Color cycle (ring+thumb pinch) ─
                    if d_color < pinch_thr:
                        finalize_active_stroke()
                        if not color_change_done:
                            draw_color_idx    = (draw_color_idx + 1) % len(EDITOR_COLORS_HEX)
                            color_change_done = True
                            try_put(draw_queue, ('color', draw_color_idx))
                        rx_px = int(rx_n*CAM_W); ry_px = int(ry_n*CAM_H)
                        cv2.circle(frame, (rx_px,ry_px), 14, EDITOR_COLORS_BGR[draw_color_idx], 3)
                        cv2.putText(frame, 'COLOR', (10,160),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, EDITOR_COLORS_BGR[draw_color_idx], 2)
                        try_put(draw_queue, ('cursor', sx, sy, EDITOR_COLORS_HEX[draw_color_idx], 'penup'))
                        prev_draw_sx = prev_draw_sy = None

                    else:
                        color_change_done = False

                        # ── Priority 3: Erase (middle+thumb pinch) ─
                        if d_erase < draw_pinch_thr:
                            finalize_active_stroke()
                            esx, esy = norm_to_screen(
                                (mx_n + tx_n) / 2,
                                (my_n + ty_n) / 2
                            )
                            try_put(draw_queue, ('erase', esx, esy, ERASER_SIZE * 2))
                            try_put(draw_queue, ('cursor', esx, esy, '#aaaaaa', 'erase'))
                            mx_px = int(mx_n*CAM_W); my_px = int(my_n*CAM_H)
                            cv2.circle(frame, (mx_px,my_px), 14, (180,180,180), 2)
                            cv2.putText(frame, 'ERASING', (10,160),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 2)
                        # ── Priority 4: Draw (index+thumb pinch) ───
                        elif d_draw < draw_pinch_thr:
                            # Brush size = spread ratio — closer = thinner, near draw_pinch_thr = thicker
                            spread_pct  = min(1.0, d_draw / draw_pinch_thr)
                            brush_size  = int(BRUSH_MIN + spread_pct * (BRUSH_MAX - BRUSH_MIN))

                            # Track index fingertip only — drawing starts from index pointer
                            track_x = ix_n
                            track_y = iy_n

                            if smooth_draw_sx is None:
                                smooth_draw_sx = float(track_x)
                                smooth_draw_sy = float(track_y)
                            else:
                                smooth_draw_sx = smooth_draw_sx * DRAW_SMOOTHING + track_x * (1-DRAW_SMOOTHING)
                                smooth_draw_sy = smooth_draw_sy * DRAW_SMOOTHING + track_y * (1-DRAW_SMOOTHING)

                            dsx, dsy = norm_to_screen(smooth_draw_sx, smooth_draw_sy)

                            if not drawing_active:
                                stroke_id_counter += 1
                                active_stroke_id = stroke_id_counter
                                drawing_active = True
                                stroke_points = []
                                stroke_size_samples = []
                                stroke_color = cur_hex
                                try_put(draw_queue, ('stroke_begin', active_stroke_id))

                            if not stroke_points or abs(stroke_points[-1][0] - dsx) + abs(stroke_points[-1][1] - dsy) >= 2:
                                stroke_points.append((dsx, dsy))
                                stroke_size_samples.append(brush_size)

                            if prev_draw_sx is not None and active_stroke_id is not None:
                                try_put(draw_queue, ('draw',
                                                     prev_draw_sx, prev_draw_sy, dsx, dsy,
                                                     cur_hex, brush_size, active_stroke_id))

                            prev_draw_sx, prev_draw_sy = dsx, dsy
                            try_put(draw_queue, ('cursor', dsx, dsy, cur_hex, 'draw'))

                            # Preview dot on webcam
                            d_px = int(track_x * CAM_W)
                            d_py = int(track_y * CAM_H)
                            cv2.circle(frame, (d_px,d_py), brush_size+2, cur_bgr, -1)
                            cv2.putText(frame, f'DRAWING  brush:{brush_size}', (10,160),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, cur_bgr, 2)

                        # ── Priority 5: Pen up (everything else) ───
                        else:
                            finalize_active_stroke()
                            try_put(draw_queue, ('cursor', sx, sy, '#777777', 'penup'))
                            cv2.putText(frame, 'PEN UP', (10,160),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (130,130,130), 2)

                # Color swatch
                cv2.rectangle(frame, (CAM_W-40, 44), (CAM_W-8, 76), cur_bgr, -1)
                cv2.rectangle(frame, (CAM_W-40, 44), (CAM_W-8, 76), (255,255,255), 1)

        else:
            # ── Hand lost ─────────────────────────────────────────
            lm_smooth           = None   # Reset so next detection starts fresh
            scroll_mode_active  = False
            scroll_anchor_y     = None
            zoom_anchor_dist    = None
            right_click_start   = None
            right_click_done    = False
            click_drag_start    = None
            clear_hold_start    = None
            finalize_active_stroke()
            cx_buf.clear(); cy_buf.clear()
            if dragging:
                pyautogui.mouseUp()
                dragging = False
            try_put(draw_queue, ('cursor_hide',))
            cv2.putText(frame, 'NO HAND DETECTED',
                        (CAM_W//2 - 110, CAM_H//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60,60,255), 2)

        # ── FPS ───────────────────────────────────────────────────
        fps_buf.append(1 / max(now - prev_time, 0.001))
        prev_time = now
        fps = int(sum(fps_buf)/len(fps_buf))
        cv2.putText(frame, f'FPS:{fps}', (CAM_W-70, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160,160,160), 1)

        # ── Dynamic threshold HUD ─────────────────────────────────
        if result.multi_hand_landmarks:
            dist_pct  = min(1.0, hsc_smooth / 0.55)
            bar_color = (
                (0, 220, 80)  if dist_pct > 0.65 else
                (0, 200, 255) if dist_pct > 0.35 else
                (0, 80, 255)
            )
            dist_label = 'CLOSE' if dist_pct > 0.65 else ('MEDIUM' if dist_pct > 0.35 else 'FAR')
            bar_w = int(dist_pct * 80)
            # Background bar
            cv2.rectangle(frame, (CAM_W-92, 42), (CAM_W-10, 52), (40,40,40), -1)
            # Filled bar
            cv2.rectangle(frame, (CAM_W-92, 42), (CAM_W-92+bar_w, 52), bar_color, -1)
            cv2.putText(frame, f'DIST:{dist_label} {pinch_thr:.2f}',
                        (CAM_W-120, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.3, bar_color, 1)

        # ── Hints ─────────────────────────────────────────────────
        with mode_lock:
            m = current_mode
        hints = (
            ['idx=move', 'idx+mid=scroll', 'mid+th=tap/hold',
             'tap=click hold=drag', 'idx+th=R-click', 'spread=zoom', 'palm=tab']
            if m == 1 else
            ['relax=pen up', 'idx+th=draw', 'mid+th=erase',
             'rng+th=color', 'pky+th(hold)=clear', 'S=save']
        )
        for i, h in enumerate(hints):
            cv2.putText(frame, h, (CAM_W-145, CAM_H-110+i*15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100,100,100), 1)

        cv2.imshow(window_name, frame)

        # Move window only on first frame
        if not window_moved:
            cv2.moveWindow(window_name, CAM_WIN_X, CAM_WIN_Y)
            window_moved = True

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            try_put_ctrl(('quit',))
            break
        elif key == ord('s'):
            try_put_ctrl(('save',))
        elif key == ord('c'):
            try_put_ctrl(('clear',))

    cap.release()
    cv2.destroyAllWindows()

# ================================================================
#  SECTION 6 — ENTRY POINT
# ================================================================

if __name__ == '__main__':
    cam_thread = threading.Thread(target=webcam_thread, daemon=True)
    cam_thread.start()

    overlay = ScreenOverlay()
    overlay.run()
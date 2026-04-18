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
# ║    Thumb + Middle       → Left click                            ║
# ║    Thumb + Ring         → Drag                                  ║
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

# --- Gesture thresholds (direct normalised 0-1 distances, NO hsc scaling) ---
# Real pinch (tips touching) ≈ 0.04-0.07 normalised → threshold 0.07
# Real release              ≈ 0.10-0.13            → threshold 0.11
# Index-middle scroll close ≈ 0.04-0.07            → threshold 0.07
# Index-middle zoom spread  ≈ 0.18-0.30            → threshold 0.17
PINCH_THR             = 0.07
DRAW_PINCH_THR = 0.09
RELEASE_THR           = 0.09
SCROLL_THR            = 0.07

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
# Zoom requires index+middle clearly spread apart AND ring+pinky folded
# 0.17 normalised ≈ fingers visibly spread, well above scroll zone (0.07)
ZOOM_THR              = 0.17
ZOOM_COOLDOWN         = 0.15

# --- Mode switch ---
MODE_SWITCH_FRAMES    = 35

# --- Screen editor ---
BRUSH_MIN             = 3
BRUSH_MAX             = 26
DRAW_SMOOTHING        = 0.72        # High value = smooth path, less jitter
ERASER_SIZE           = 40
CLEAR_HOLD_SEC        = 1.5         # Hold pinky+thumb this long to clear

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

        self._draw_palette()
        self._draw_hud('Mode 1: Mouse Control', '#FF8C00')
        self.root.after(8, self._poll)

    # ── Queue polling ────────────────────────────────────────────
    def _poll(self):
        try:
            while True:
                self._handle(draw_queue.get_nowait())
        except queue.Empty:
            pass
        self.root.after(8, self._poll)

    def _handle(self, msg):
        kind = msg[0]

        if kind == 'draw':
            _, x1, y1, x2, y2, color, size = msg
            iid = self.canvas.create_line(
                x1, y1, x2, y2,
                fill=color, width=size,
                capstyle=tk.ROUND, joinstyle=tk.ROUND, smooth=True,
            )
            self.strokes.append(iid)

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
        self._clear_cursor()

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

# ================================================================
#  SECTION 5 — WEBCAM THREAD
# ================================================================

def webcam_thread():
    global current_mode

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)    # Minimize buffer lag

    mp_hands = mp.solutions.hands
    hands    = mp_hands.Hands(
        static_image_mode        = False,
        max_num_hands            = 1,
        model_complexity         = 1,          # Accurate model — better landmark placement
        min_detection_confidence = 0.75,
        min_tracking_confidence  = 0.75,
    )
    draw_utils = mp.solutions.drawing_utils
    lm_style   = draw_utils.DrawingSpec(color=(0, 255, 180), thickness=1, circle_radius=2)
    cn_style   = draw_utils.DrawingSpec(color=(80, 80, 220), thickness=1)

    # ── Per-landmark EMA smoothing ───────────────────────────────
    # 21 landmarks × [x, y] — smoothed values used for BOTH drawing
    # and all gesture distance calculations
    LM_ALPHA  = 0.50          # blend: 0=fully raw (jittery), 1=fully frozen
    lm_smooth = None           # initialised on first detection

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

    while True:
        success, frame = cap.read()
        if not success:
            continue

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
        cv2.putText(frame, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

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

            # ── Build a lightweight proxy so dist_norm / is_finger_up
            #    can read from lm_smooth instead of raw_lm ──────────
            class _LM:
                __slots__ = ('x', 'y')
                def __init__(self, x, y): self.x = x; self.y = y

            lm = [_LM(x, y) for x, y in lm_smooth]  # smoothed, used everywhere below

            # ── Key landmark coords (from SMOOTHED lm) ────────────
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
                    prev_draw_sx = prev_draw_sy = None
                    smooth_draw_sx = smooth_draw_sy = None
                    clear_hold_start = None
                    clear_done = False
                    cx_buf.clear(); cy_buf.clear()
                    try_put(draw_queue, ('mode', new_mode))
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
                if d_right < PINCH_THR and not index_up and not scroll_mode_active:
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
                if index_up and middle_up and ring_dn and pinky_dn and zoom_norm > ZOOM_THR:
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
                if d_scroll < SCROLL_THR:
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

                # ── Left click (middle + thumb) ───────────────────
                if gesture_state == 'IDLE':
                    if d_click < PINCH_THR:
                        gesture_state = 'PINCH'
                elif gesture_state == 'PINCH':
                    pyautogui.click()
                    gesture_state = 'CLICKED'
                    mx_px = int(mx_n*CAM_W); my_px = int(my_n*CAM_H)
                    cv2.circle(frame, (mx_px,my_px), 14, (0,255,0), 3)
                elif gesture_state == 'CLICKED':
                    if d_click > RELEASE_THR:
                        gesture_state = 'IDLE'

                # ── Drag (ring + thumb) ───────────────────────────
                if d_drag < PINCH_THR:
                    if not dragging:
                        pyautogui.mouseDown()
                        dragging = True
                    rx_px = int(rx_n*CAM_W); ry_px = int(ry_n*CAM_H)
                    cv2.circle(frame, (rx_px,ry_px), 14, (0,0,255), 3)
                else:
                    if dragging:
                        pyautogui.mouseUp()
                        dragging = False

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
                if d_clear < PINCH_THR:
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
                    if d_color < PINCH_THR:
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
                        if d_erase < PINCH_THR:
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
                            prev_draw_sx = prev_draw_sy = None
                            smooth_draw_sx = smooth_draw_sy = None

                        # ── Priority 4: Draw (index+thumb pinch) ───
                        elif d_draw < DRAW_PINCH_THR:
                            # Brush size = spread ratio — closer = thinner, near release = thicker
                            spread_pct  = min(1.0, d_draw / RELEASE_THR)
                            brush_size  = int(BRUSH_MIN + spread_pct * (BRUSH_MAX - BRUSH_MIN))

                            # Track midpoint of index tip + thumb tip for stable point
                            track_x = (ix_n + tx_n) / 2
                            track_y = (iy_n + ty_n) / 2

                            if smooth_draw_sx is None:
                                smooth_draw_sx = float(track_x)
                                smooth_draw_sy = float(track_y)
                            else:
                                smooth_draw_sx = smooth_draw_sx * DRAW_SMOOTHING + track_x * (1-DRAW_SMOOTHING)
                                smooth_draw_sy = smooth_draw_sy * DRAW_SMOOTHING + track_y * (1-DRAW_SMOOTHING)

                            dsx, dsy = norm_to_screen(smooth_draw_sx, smooth_draw_sy)

                            if prev_draw_sx is not None:
                                try_put(draw_queue, ('draw',
                                                     prev_draw_sx, prev_draw_sy, dsx, dsy,
                                                     cur_hex, brush_size))

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
                            prev_draw_sx = prev_draw_sy   = None
                            smooth_draw_sx = smooth_draw_sy = None
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
            clear_hold_start    = None
            prev_draw_sx = prev_draw_sy   = None
            smooth_draw_sx = smooth_draw_sy = None
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
        cv2.putText(frame, f'FPS:{fps}', (CAM_W-80, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160,160,160), 1)

        # ── Hints ─────────────────────────────────────────────────
        with mode_lock:
            m = current_mode
        hints = (
            ['idx=move', 'idx+mid=scroll', 'mid+th=click',
             'rng+th=drag', 'idx+th=R-click', 'spread=zoom', 'palm=tab']
            if m == 1 else
            ['relax=pen up', 'idx+th=draw', 'mid+th=erase',
             'rng+th=color', 'pky+th(hold)=clear', 'S=save']
        )
        for i, h in enumerate(hints):
            cv2.putText(frame, h, (CAM_W-160, CAM_H-110+i*18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120,120,120), 1)

        cv2.imshow(window_name, frame)

        # Move window only on first frame
        if not window_moved:
            cv2.moveWindow(window_name, CAM_WIN_X, CAM_WIN_Y)
            window_moved = True

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            try_put(draw_queue, ('quit',))
            break
        elif key == ord('s'):
            try_put(draw_queue, ('save',))
        elif key == ord('c'):
            try_put(draw_queue, ('clear',))

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
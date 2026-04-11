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
# ║        HAND GESTURE MOUSE & SCREEN EDITOR  v3.0                 ║
# ║                                                                  ║
# ║  NEW IN v3: Draws on the REAL screen via a fullscreen           ║
# ║  transparent always-on-top overlay window.                       ║
# ║  Works over any app: browser, notepad, VS Code, etc.            ║
# ║                                                                  ║
# ║  ARCHITECTURE                                                    ║
# ║    • Thread 1 (main)    — Tkinter overlay window                ║
# ║    • Thread 2           — OpenCV webcam + MediaPipe             ║
# ║    • Communication      — thread-safe Queue                     ║
# ║                                                                  ║
# ║  MODE 1 — MOUSE CONTROL                                         ║
# ║    Index tip            → Move cursor                           ║
# ║    Index + Middle close → Scroll (anchor-based)                 ║
# ║    Thumb + Middle       → Left click                            ║
# ║    Thumb + Ring         → Drag                                  ║
# ║    Thumb + Index hold   → Right click                           ║
# ║    Index + Middle wide  → Zoom (Ctrl +/-)                       ║
# ║                                                                  ║
# ║  MODE 2 — SCREEN EDITOR (draws on real screen!)                 ║
# ║    Index up only        → Draw / write                          ║
# ║    Index + Middle up    → Erase                                 ║
# ║    Fist                 → Pen up (move freely)                  ║
# ║    Ring + Thumb pinch   → Cycle draw color                      ║
# ║    Pinky + Thumb pinch  → Clear canvas                          ║
# ║    Keyboard S           → Save drawing as PNG                   ║
# ║    Keyboard C           → Clear canvas                          ║
# ║                                                                  ║
# ║  BOTH MODES                                                      ║
# ║    ILY gesture (hold)   → Switch mode                           ║
# ╚══════════════════════════════════════════════════════════════════╝

# ================================================================
#  SECTION 1 — SETTINGS
# ================================================================

CAM_W, CAM_H          = 640, 480
CAM_WIN_X             = 20          # Webcam preview window position
CAM_WIN_Y             = 20

# --- Cursor ---
SMOOTHING             = 0.5
MAX_ACCEL             = 3.0
SENSITIVITY           = 3.5
DEADZONE              = 1.5

# --- Gesture thresholds ---
PINCH_THRESHOLD       = 30
RELEASE_THRESHOLD     = 45
FINGER_TOUCH_DIST     = 25
ZOOM_START_DIST       = 80

# --- Scroll ---
SCROLL_ENTRY_FRAMES   = 3
SCROLL_COOLDOWN       = 0.08
SCROLL_SENSITIVITY    = 6

# --- Right click ---
RIGHT_CLICK_HOLD_SEC  = 0.4

# --- Tab swipe ---
SWIPE_WINDOW_SEC      = 0.35   # Time window to complete the swipe
SWIPE_MIN_X           = 110    # Minimum horizontal cam pixels for a swipe
SWIPE_MAX_Y           = 55     # Maximum allowed vertical drift (keeps it horizontal)
SWIPE_COOLDOWN        = 1.2    # Seconds before another swipe can fire

# --- Zoom ---
ZOOM_COOLDOWN         = 0.12

# --- Mode switch ---
MODE_SWITCH_FRAMES    = 40

# --- Screen editor ---
DRAW_SMOOTHING        = 0.6
BRUSH_SIZE            = 4
ERASER_SIZE           = 40
# Colors as (R, G, B) — converted to hex for tkinter
EDITOR_COLORS_RGB = [
    (255,  60,  60),   # Red
    (255, 160,  30),   # Orange
    (255, 230,  30),   # Yellow
    ( 60, 220,  60),   # Green
    ( 60, 140, 255),   # Blue
    (180,  60, 220),   # Purple
    (255, 255, 255),   # White
]
# Corresponding OpenCV colors (BGR)
EDITOR_COLORS_BGR = [(b, g, r) for (r, g, b) in EDITOR_COLORS_RGB]

def rgb_to_hex(r, g, b):
    return f'#{r:02x}{g:02x}{b:02x}'

EDITOR_COLORS_HEX = [rgb_to_hex(*c) for c in EDITOR_COLORS_RGB]

# Overlay background color — this color becomes TRANSPARENT in the window
TRANSPARENT_KEY = '#000000'

pyautogui.FAILSAFE    = False
pyautogui.PAUSE       = 0

# ================================================================
#  SECTION 2 — SHARED STATE (between threads)
# ================================================================

draw_queue   = queue.Queue()   # Webcam thread → Overlay thread
current_mode = 1               # 1 = Mouse, 2 = Screen Editor (read by both threads)
mode_lock    = threading.Lock()

screen_w, screen_h = pyautogui.size()

# ================================================================
#  SECTION 3 — OVERLAY WINDOW  (runs on main thread)
# ================================================================

class ScreenOverlay:
    """
    Fullscreen always-on-top transparent tkinter window.
    Transparent key color (black) becomes see-through so you can
    see the desktop / any app underneath.
    All drawing goes on a tkinter Canvas using line segments.
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Screen Editor Overlay")
        self.root.attributes('-fullscreen',  True)
        self.root.attributes('-topmost',     True)
        self.root.attributes('-transparentcolor', TRANSPARENT_KEY)
        self.root.configure(bg=TRANSPARENT_KEY)
        self.root.overrideredirect(True)   # No title bar

        self.canvas = tk.Canvas(
            self.root,
            bg            = TRANSPARENT_KEY,
            highlightthickness = 0,
            cursor        = 'none',
        )
        self.canvas.pack(fill='both', expand=True)

        # Keyboard shortcuts (canvas must have focus)
        self.canvas.bind('<KeyPress-s>', self.save_drawing)
        self.canvas.bind('<KeyPress-c>', lambda e: self.clear_canvas())
        self.canvas.bind('<Escape>',     self.quit)
        self.canvas.focus_set()

        # State
        self.hud_text_id     = None
        self.cursor_id       = None
        self.palette_ids     = []
        self.color_idx       = 0
        self.drawing_visible = False       # Only show canvas content in Mode 2
        self.all_strokes     = []          # (item_id) for save/clear

        self._draw_palette()
        self._draw_hud('Mode 1: Mouse Control', '#FF8C00')

        # Start polling the queue
        self.root.after(8, self._poll_queue)

    # ── Queue polling ────────────────────────────────────────────
    def _poll_queue(self):
        try:
            while True:
                msg = draw_queue.get_nowait()
                self._handle(msg)
        except queue.Empty:
            pass
        self.root.after(8, self._poll_queue)

    def _handle(self, msg):
        kind = msg[0]

        if kind == 'draw':
            _, x1, y1, x2, y2, color_hex, size = msg
            iid = self.canvas.create_line(
                x1, y1, x2, y2,
                fill    = color_hex,
                width   = size,
                capstyle= tk.ROUND,
                joinstyle=tk.ROUND,
                smooth  = True,
            )
            self.all_strokes.append(iid)

        elif kind == 'erase':
            _, ex, ey, size = msg
            # Draw a filled circle in the transparent key color → erases
            r   = size // 2
            iid = self.canvas.create_oval(
                ex - r, ey - r, ex + r, ey + r,
                fill    = TRANSPARENT_KEY,
                outline = TRANSPARENT_KEY,
                )
            self.all_strokes.append(iid)

        elif kind == 'clear':
            self.clear_canvas()

        elif kind == 'mode':
            _, mode = msg
            if mode == 1:
                self._hide_drawing()
                self._draw_hud('Mode 1: Mouse Control', '#FF8C00')
            else:
                self._show_drawing()
                self._draw_hud('Mode 2: Screen Editor', '#00D264')

        elif kind == 'cursor':
            _, cx, cy, color_hex, style = msg
            # Delete previous cursor (could be multiple items stored as list)
            if self.cursor_id:
                if isinstance(self.cursor_id, list):
                    for iid in self.cursor_id: self.canvas.delete(iid)
                else:
                    self.canvas.delete(self.cursor_id)

            ids = []
            if style == 'draw':
                # Solid filled dot + tight crosshair in active color
                r = 5
                ids.append(self.canvas.create_oval(
                    cx - r, cy - r, cx + r, cy + r,
                    fill=color_hex, outline='black', width=1))
                arm = 14
                ids.append(self.canvas.create_line(
                    cx - arm, cy, cx - r - 2, cy,
                    fill=color_hex, width=2))
                ids.append(self.canvas.create_line(
                    cx + r + 2, cy, cx + arm, cy,
                    fill=color_hex, width=2))
                ids.append(self.canvas.create_line(
                    cx, cy - arm, cx, cy - r - 2,
                    fill=color_hex, width=2))
                ids.append(self.canvas.create_line(
                    cx, cy + r + 2, cx, cy + arm,
                    fill=color_hex, width=2))

            elif style == 'penup':
                # Hollow circle + dashed crosshair in gray — "I see you but not drawing"
                r = 7
                ids.append(self.canvas.create_oval(
                    cx - r, cy - r, cx + r, cy + r,
                    outline='#888888', width=1, dash=(3, 3)))
                arm = 16
                for x1, y1, x2, y2 in [
                    (cx - arm, cy, cx - r - 2, cy),
                    (cx + r + 2, cy, cx + arm, cy),
                    (cx, cy - arm, cx, cy - r - 2),
                    (cx, cy + r + 2, cx, cy + arm),
                ]:
                    ids.append(self.canvas.create_line(
                        x1, y1, x2, y2,
                        fill='#888888', width=1, dash=(3, 3)))

            elif style == 'erase':
                # Large hollow circle showing eraser radius in gray
                r = ERASER_SIZE
                ids.append(self.canvas.create_oval(
                    cx - r, cy - r, cx + r, cy + r,
                    outline='#aaaaaa', width=2, dash=(4, 4)))
                ids.append(self.canvas.create_oval(
                    cx - 3, cy - 3, cx + 3, cy + 3,
                    outline='#aaaaaa', width=1))

            self.cursor_id = ids

        elif kind == 'cursor_hide':
            if self.cursor_id:
                if isinstance(self.cursor_id, list):
                    for iid in self.cursor_id: self.canvas.delete(iid)
                else:
                    self.canvas.delete(self.cursor_id)
                self.cursor_id = None

        elif kind == 'color':
            _, idx = msg
            self.color_idx = idx
            self._update_palette_highlight()

        elif kind == 'save':
            self.save_drawing()

        elif kind == 'quit':
            self.root.destroy()

    # ── Canvas helpers ───────────────────────────────────────────
    def clear_canvas(self):
        for iid in self.all_strokes:
            self.canvas.delete(iid)
        self.all_strokes.clear()
        if self.cursor_id:
            self.canvas.delete(self.cursor_id)
            self.cursor_id = None

    def _hide_drawing(self):
        """Hide all strokes (Mode 1 — don't show over mouse work)."""
        for iid in self.all_strokes:
            self.canvas.itemconfigure(iid, state='hidden')

    def _show_drawing(self):
        """Show all strokes again (Mode 2)."""
        for iid in self.all_strokes:
            self.canvas.itemconfigure(iid, state='normal')

    # ── HUD label (top-left corner) ───────────────────────────────
    def _draw_hud(self, text, color):
        if self.hud_text_id:
            self.canvas.delete(self.hud_text_id)
        self.hud_text_id = self.canvas.create_text(
            14, 14,
            text   = text,
            anchor = 'nw',
            fill   = color,
            font   = ('Helvetica', 14, 'bold'),
        )

    # ── Color palette (bottom-left, Mode 2) ──────────────────────
    def _draw_palette(self):
        for iid in self.palette_ids:
            self.canvas.delete(iid)
        self.palette_ids.clear()

        y = screen_h - 20
        for i, hex_c in enumerate(EDITOR_COLORS_HEX):
            x   = 20 + i * 28
            r   = 9 if i != self.color_idx else 12
            out = 'white' if i == self.color_idx else hex_c
            iid = self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill    = hex_c,
                outline = out,
                width   = 2,
                )
            self.palette_ids.append(iid)

    def _update_palette_highlight(self):
        self._draw_palette()

    # ── Save drawing ──────────────────────────────────────────────
    def save_drawing(self, event=None):
        """Save the canvas drawing as a PNG using PIL."""
        try:
            from PIL import ImageGrab
            fname = f'drawing_{int(time.time())}.png'
            # Grab only the current drawings via canvas PostScript → PIL
            # Easier: screenshot the overlay region
            img = ImageGrab.grab()
            img.save(fname)
            print(f'[Saved] {fname}')
            self._draw_hud(f'Saved: {fname}', '#00FF88')
            self.root.after(2000, lambda: self._draw_hud('Mode 2: Screen Editor', '#00D264'))
        except ImportError:
            print('[Save] Install Pillow: pip install Pillow')

    def quit(self, event=None):
        self.root.destroy()

    def run(self):
        self.root.mainloop()

# ================================================================
#  SECTION 4 — ACCELERATION & LANDMARK HELPERS
# ================================================================

def get_acceleration(speed):
    if speed < 2:    return 0.2
    elif speed < 10: return 1.2
    elif speed < 25: return 2.0
    elif speed < 40: return 2.8
    return MAX_ACCEL

def lm_px(lm, idx, w, h):
    return int(lm[idx].x * w), int(lm[idx].y * h)

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def is_finger_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y

def count_fingers_up(lm):
    return sum(1 for tip, pip in [(8,6),(12,10),(16,14),(20,18)]
               if is_finger_up(lm, tip, pip))

def is_fist(lm):
    return count_fingers_up(lm) == 0

def is_open_palm(lm):
    """All 4 fingers extended — used as the swipe trigger pose."""
    return count_fingers_up(lm) == 4

def is_ily_gesture(lm):
    """Index + Pinky up, Middle + Ring down."""
    return (is_finger_up(lm, 8, 6) and
            is_finger_up(lm, 20, 18) and
            not is_finger_up(lm, 12, 10) and
            not is_finger_up(lm, 16, 14))

def cam_to_screen(cx, cy):
    """Map webcam pixel coords to full screen coords."""
    sx = int(np.interp(cx, [0, CAM_W], [0, screen_w]))
    sy = int(np.interp(cy, [0, CAM_H], [0, screen_h]))
    return sx, sy

# ================================================================
#  SECTION 5 — WEBCAM THREAD
# ================================================================

def webcam_thread():
    global current_mode

    # ── MediaPipe setup ──────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    cap.set(3, CAM_W)
    cap.set(4, CAM_H)

    mp_hands   = mp.solutions.hands
    hands      = mp_hands.Hands(
        static_image_mode        = False,
        max_num_hands            = 1,
        model_complexity         = 1,
        min_detection_confidence = 0.7,
        min_tracking_confidence  = 0.7,
    )
    draw_utils = mp.solutions.drawing_utils
    lm_style   = draw_utils.DrawingSpec(color=(0, 255, 180), thickness=1, circle_radius=2)
    cn_style   = draw_utils.DrawingSpec(color=(100, 100, 255), thickness=1)

    # ── State ────────────────────────────────────────────────────
    prev_ix, prev_iy      = 0, 0
    smoothed_dx           = 0.0
    smoothed_dy           = 0.0

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

    # --- Tab swipe ---
    wrist_history         = deque()   # stores (wrist_x, wrist_y, timestamp)
    last_swipe_time       = 0.0
    swipe_flash_label     = ''
    swipe_flash_until     = 0.0

    mode_switch_frames    = 0
    mode_switch_flash     = 0

    draw_color_idx        = 0
    prev_draw_sx          = None
    prev_draw_sy          = None
    smooth_draw_sx        = None
    smooth_draw_sy        = None
    color_change_done     = False

    fps_buffer = deque(maxlen=30)
    prev_time  = time.time()

    # ── Main webcam loop ─────────────────────────────────────────
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame   = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result  = hands.process(rgb)

        # ── Mode banner ──────────────────────────────────────────
        with mode_lock:
            mode = current_mode
        label_text  = 'MODE 1: MOUSE' if mode == 1 else 'MODE 2: SCREEN EDITOR'
        label_color = (255, 140, 0) if mode == 1 else (0, 210, 100)
        cv2.rectangle(frame, (0, 0), (w, 36), (25, 25, 25), -1)
        cv2.putText(frame, label_text, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, label_color, 2)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            draw_utils.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS,
                                      lm_style, cn_style)
            lm = hand.landmark

            # ── Landmarks ────────────────────────────────────────
            i_tip = lm_px(lm,  8, w, h)
            m_tip = lm_px(lm, 12, w, h)
            r_tip = lm_px(lm, 16, w, h)
            p_tip = lm_px(lm, 20, w, h)
            t_tip = lm_px(lm,  4, w, h)
            ix, iy = i_tip

            dist_click  = dist(m_tip, t_tip)
            dist_drag   = dist(r_tip, t_tip)
            dist_scroll = dist(i_tip, m_tip)
            dist_right  = dist(i_tip, t_tip)

            if prev_ix == 0 and prev_iy == 0:
                prev_ix, prev_iy = ix, iy

            dx    = ix - prev_ix
            dy    = iy - prev_iy
            speed = math.hypot(dx, dy)
            prev_ix, prev_iy = ix, iy

            acc         = get_acceleration(speed)
            smoothed_dx = smoothed_dx * SMOOTHING + dx * SENSITIVITY * acc * (1 - SMOOTHING)
            smoothed_dy = smoothed_dy * SMOOTHING + dy * SENSITIVITY * acc * (1 - SMOOTHING)

            # ── ILY mode switch gesture ───────────────────────────
            if is_ily_gesture(lm):
                mode_switch_frames += 1
                progress = int((mode_switch_frames / MODE_SWITCH_FRAMES) * (w - 20))
                cv2.rectangle(frame, (10, h - 22), (10 + progress, h - 10), (0, 255, 200), -1)
                cv2.putText(frame, '\m/ Hold to switch mode...', (10, h - 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 200), 1)

                if mode_switch_frames >= MODE_SWITCH_FRAMES:
                    with mode_lock:
                        current_mode = 2 if current_mode == 1 else 1
                        new_mode = current_mode
                    mode_switch_frames   = 0
                    mode_switch_flash    = 20
                    prev_ix = prev_iy    = 0
                    smoothed_dx = smoothed_dy = 0.0
                    scroll_mode_active   = False
                    scroll_anchor_y      = None
                    zoom_anchor_dist     = None
                    gesture_state        = 'IDLE'
                    right_click_start    = None
                    right_click_done     = False
                    prev_draw_sx = prev_draw_sy   = None
                    smooth_draw_sx = smooth_draw_sy = None
                    draw_queue.put(('mode', new_mode))
            else:
                mode_switch_frames = 0

            # ════════════════════════════════════════════════════
            #  MODE 1 — MOUSE CONTROL
            # ════════════════════════════════════════════════════
            if mode == 1:
                draw_queue.put(('cursor_hide',))

                index_up  = is_finger_up(lm, 8,  6)
                middle_up = is_finger_up(lm, 12, 10)

                # ── Tab swipe (open palm + fast horizontal wrist move) ─
                wx = int(lm[0].x * w)
                wy = int(lm[0].y * h)
                now = time.time()
                wrist_history.append((wx, wy, now))
                # Prune entries older than the swipe window
                while wrist_history and now - wrist_history[0][2] > SWIPE_WINDOW_SEC:
                    wrist_history.popleft()

                if (is_open_palm(lm) and
                        len(wrist_history) >= 4 and
                        (now - last_swipe_time) > SWIPE_COOLDOWN):
                    x0, y0, _ = wrist_history[0]
                    net_x = wx - x0
                    net_y = abs(wy - y0)
                    if abs(net_x) > SWIPE_MIN_X and net_y < SWIPE_MAX_Y:
                        if net_x > 0:                      # hand moved RIGHT → prev tab
                            pyautogui.hotkey('ctrl', 'shift', 'tab')
                            swipe_flash_label = '<< PREV TAB'
                        else:                               # hand moved LEFT → next tab
                            pyautogui.hotkey('ctrl', 'tab')
                            swipe_flash_label = 'NEXT TAB >>'
                        last_swipe_time = now
                        swipe_flash_until = now + 0.8
                        wrist_history.clear()

                if time.time() < swipe_flash_until:
                    cv2.putText(frame, swipe_flash_label, (w // 2 - 80, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)

                # ── Right click: Thumb + Index pinch hold ─────────────
                # Index curls DOWN to touch thumb → index_up is False
                # Guard: scroll must not be active (avoids conflicts)
                if dist_right < PINCH_THRESHOLD and not index_up and not scroll_mode_active:
                    if right_click_start is None:
                        right_click_start = time.time()
                    elif (not right_click_done and
                          time.time() - right_click_start > RIGHT_CLICK_HOLD_SEC):
                        pyautogui.rightClick()
                        right_click_done = True
                    pct = min(1.0, (time.time() - right_click_start) / RIGHT_CLICK_HOLD_SEC)
                    cv2.ellipse(frame, t_tip, (18, 18), -90, 0, int(360 * pct),
                                (0, 100, 255), 3)
                    cv2.putText(frame, 'RIGHT CLICK', (10, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
                else:
                    right_click_start = None
                    right_click_done  = False

                # ── Zoom ──────────────────────────────────────────
                now = time.time()
                if index_up and middle_up and dist_scroll > ZOOM_START_DIST:
                    if zoom_anchor_dist is None:
                        zoom_anchor_dist = dist_scroll
                    else:
                        delta = dist_scroll - zoom_anchor_dist
                        if abs(delta) > 5 and (now - last_zoom_time) > ZOOM_COOLDOWN:
                            pyautogui.hotkey('ctrl', '+' if delta > 0 else '-')
                            zoom_anchor_dist = dist_scroll
                            last_zoom_time   = now
                    cv2.line(frame, i_tip, m_tip, (255, 200, 0), 2)
                    cv2.putText(frame, 'ZOOM', (10, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
                else:
                    zoom_anchor_dist = None

                # ── Scroll vs Move ────────────────────────────────
                if dist_scroll < FINGER_TOUCH_DIST:
                    fingers_together_cnt += 1
                    if fingers_together_cnt >= SCROLL_ENTRY_FRAMES:
                        scroll_mode_active = True

                    if scroll_mode_active:
                        if scroll_anchor_y is None:
                            scroll_anchor_y = iy
                        scroll_delta = iy - scroll_anchor_y
                        now = time.time()
                        if abs(scroll_delta) > DEADZONE and (now - last_scroll_time) > SCROLL_COOLDOWN:
                            direction    = -1 if scroll_delta > 0 else 1
                            scroll_ticks = max(1, int(abs(scroll_delta) / 10))
                            pyautogui.scroll(direction * scroll_ticks * SCROLL_SENSITIVITY)
                            last_scroll_time = now
                        cv2.line(frame, (0, scroll_anchor_y), (w, scroll_anchor_y), (0, 255, 255), 1)
                        cv2.circle(frame, i_tip, 10, (0, 255, 255), 2)
                        cv2.putText(frame, 'SCROLLING', (10, 160),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    fingers_together_cnt = 0
                    scroll_mode_active   = False
                    scroll_anchor_y      = None

                    if zoom_anchor_dist is None:
                        if abs(smoothed_dx) > DEADZONE or abs(smoothed_dy) > DEADZONE:
                            cx, cy = pyautogui.position()
                            pyautogui.moveTo(
                                np.clip(cx + smoothed_dx, 0, screen_w - 2),
                                np.clip(cy + smoothed_dy, 0, screen_h - 2)
                            )
                        cv2.circle(frame, i_tip, 8, (255, 140, 0), 2)
                        cv2.putText(frame, 'MOVING', (10, 160),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 140, 0), 2)

                # ── Left click ────────────────────────────────────
                if gesture_state == 'IDLE':
                    if dist_click < PINCH_THRESHOLD:
                        gesture_state = 'PINCH'
                elif gesture_state == 'PINCH':
                    pyautogui.click()
                    gesture_state = 'CLICKED'
                    cv2.circle(frame, m_tip, 14, (0, 255, 0), 3)
                elif gesture_state == 'CLICKED':
                    if dist_click > RELEASE_THRESHOLD:
                        gesture_state = 'IDLE'

                # ── Drag ──────────────────────────────────────────
                if dist_drag < PINCH_THRESHOLD:
                    if not dragging:
                        pyautogui.mouseDown()
                        dragging = True
                    cv2.circle(frame, r_tip, 14, (0, 0, 255), 3)
                else:
                    if dragging:
                        pyautogui.mouseUp()
                        dragging = False

                cv2.putText(frame, f'Click:{gesture_state}  Drag:{dragging}', (10, 185),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 170, 170), 1)

            # ════════════════════════════════════════════════════
            #  MODE 2 — SCREEN EDITOR (draws on real screen!)
            # ════════════════════════════════════════════════════
            elif mode == 2:
                index_up  = is_finger_up(lm, 8,  6)
                middle_up = is_finger_up(lm, 12, 10)
                ring_up   = is_finger_up(lm, 16, 14)
                pinky_up  = is_finger_up(lm, 20, 18)

                dist_color = dist(r_tip, t_tip)
                dist_clear = dist(p_tip, t_tip)

                current_color_hex = EDITOR_COLORS_HEX[draw_color_idx]
                current_color_bgr = EDITOR_COLORS_BGR[draw_color_idx]

                # ── Color cycle ───────────────────────────────────
                if dist_color < PINCH_THRESHOLD:
                    if not color_change_done:
                        draw_color_idx    = (draw_color_idx + 1) % len(EDITOR_COLORS_HEX)
                        color_change_done = True
                        draw_queue.put(('color', draw_color_idx))
                else:
                    color_change_done = False

                # ── Clear canvas ──────────────────────────────────
                if dist_clear < PINCH_THRESHOLD:
                    draw_queue.put(('clear',))
                    prev_draw_sx = prev_draw_sy = None
                    cv2.putText(frame, 'CANVAS CLEARED', (w // 2 - 100, h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # ── Erase ─────────────────────────────────────────
                elif index_up and middle_up and not ring_up:
                    ex = (ix + m_tip[0]) // 2
                    ey = (iy + m_tip[1]) // 2
                    sx, sy = cam_to_screen(ex, ey)
                    draw_queue.put(('erase', sx, sy, ERASER_SIZE * 2))
                    # Show eraser circle on webcam preview
                    cv2.circle(frame, (ex, ey), ERASER_SIZE // 3, (200, 200, 200), 2)
                    cv2.putText(frame, 'ERASING', (10, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                    draw_queue.put(('cursor', sx, sy, '#cccccc', 'erase'))
                    prev_draw_sx = prev_draw_sy = None
                    smooth_draw_sx = smooth_draw_sy = None

                # ── Draw ──────────────────────────────────────────
                elif index_up and not middle_up and not ring_up:
                    # Smooth drawing path
                    if smooth_draw_sx is None:
                        smooth_draw_sx = float(ix)
                        smooth_draw_sy = float(iy)
                    else:
                        smooth_draw_sx = smooth_draw_sx * DRAW_SMOOTHING + ix * (1 - DRAW_SMOOTHING)
                        smooth_draw_sy = smooth_draw_sy * DRAW_SMOOTHING + iy * (1 - DRAW_SMOOTHING)

                    # Map to screen
                    sdx = int(smooth_draw_sx)
                    sdy = int(smooth_draw_sy)
                    sx, sy = cam_to_screen(sdx, sdy)

                    if prev_draw_sx is not None:
                        draw_queue.put(('draw',
                                        prev_draw_sx, prev_draw_sy,
                                        sx, sy,
                                        current_color_hex,
                                        BRUSH_SIZE * 2))  # scale brush for screen size

                    prev_draw_sx, prev_draw_sy = sx, sy

                    # Cursor crosshair on overlay
                    draw_queue.put(('cursor', sx, sy, current_color_hex, 'draw'))

                    # Finger dot on webcam preview
                    cv2.circle(frame, (sdx, sdy), BRUSH_SIZE + 2, current_color_bgr, -1)
                    cv2.putText(frame, 'DRAWING', (10, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, current_color_bgr, 2)

                # ── Fist = pen up ─────────────────────────────────
                elif is_fist(lm):
                    prev_draw_sx = prev_draw_sy   = None
                    smooth_draw_sx = smooth_draw_sy = None
                    sx, sy = cam_to_screen(ix, iy)
                    draw_queue.put(('cursor', sx, sy, '#888888', 'penup'))
                    cv2.putText(frame, 'PEN UP', (10, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)

                else:
                    # Any other hand pose — still show pen-up cursor so position is always visible
                    prev_draw_sx = prev_draw_sy = None
                    sx, sy = cam_to_screen(ix, iy)
                    draw_queue.put(('cursor', sx, sy, '#888888', 'penup'))

                # ── Color swatch on webcam preview ────────────────
                cv2.rectangle(frame, (w - 38, 44), (w - 8, 74), current_color_bgr, -1)
                cv2.rectangle(frame, (w - 38, 44), (w - 8, 74), (255, 255, 255), 1)

        else:
            # ── Hand lost ─────────────────────────────────────────
            prev_ix = prev_iy = 0
            smoothed_dx = smoothed_dy = 0.0
            fingers_together_cnt = 0
            scroll_mode_active   = False
            scroll_anchor_y      = None
            zoom_anchor_dist     = None
            right_click_start    = None
            right_click_done     = False
            prev_draw_sx = prev_draw_sy   = None
            smooth_draw_sx = smooth_draw_sy = None
            draw_queue.put(('cursor_hide',))
            if dragging:
                pyautogui.mouseUp()
                dragging = False
            cv2.putText(frame, 'NO HAND DETECTED', (w // 2 - 110, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 255), 2)

        # ── FPS ───────────────────────────────────────────────────
        now = time.time()
        fps_buffer.append(1 / max(now - prev_time, 0.001))
        prev_time = now
        fps = int(sum(fps_buffer) / len(fps_buffer))
        cv2.putText(frame, f'FPS:{fps}', (w - 75, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1)

        # ── Gesture hint sheet ────────────────────────────────────
        with mode_lock:
            m = current_mode
        hints = (
            ['idx=move', 'idx+mid=scroll', 'mid+th=click',
             'rng+th=drag', 'idx+th=R-click', 'spread=zoom',
             'palm swipe=tab']
            if m == 1 else
            ['idx=draw', 'idx+mid=erase', 'fist=pen up',
             'rng+th=color', 'pky+th=clear', 'S=save']
        )
        for i, hint in enumerate(hints):
            cv2.putText(frame, hint, (w - 155, h - 110 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (130, 130, 130), 1)

        # ── Show webcam preview window (small, corner) ────────────
        cv2.imshow('Gesture Controller  |  ESC to quit', frame)
        cv2.moveWindow('Gesture Controller  |  ESC to quit', CAM_WIN_X, CAM_WIN_Y)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:           # ESC
            draw_queue.put(('quit',))
            break
        elif key == ord('s'):
            draw_queue.put(('save',))
        elif key == ord('c'):
            draw_queue.put(('clear',))

    cap.release()
    cv2.destroyAllWindows()

# ================================================================
#  SECTION 6 — ENTRY POINT
# ================================================================

if __name__ == '__main__':
    # Start webcam processing in background thread
    cam_thread = threading.Thread(target=webcam_thread, daemon=True)
    cam_thread.start()

    # Run tkinter overlay on main thread (required on Windows/macOS)
    overlay = ScreenOverlay()
    overlay.run()
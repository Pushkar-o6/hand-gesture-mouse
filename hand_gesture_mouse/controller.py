import ctypes
import importlib
import json
import pathlib
import queue
import time
from collections import deque
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

from .helpers import (
    LMPoint,
    OneEuroFilter,
    are_fingers_together,
    dist_batch,
    dist_norm,
    hand_scale,
    is_finger_up,
    is_finger_extended,
    is_fist,
    is_ily_gesture,
    is_open_palm,
    norm_to_screen,
)
from .settings import (
    BRUSH_MAX,
    BRUSH_MIN,
    CAM_H,
    CAM_W,
    CAM_WIN_X,
    CAM_WIN_Y,
    ADAPTIVE_FRAME_SKIP,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID,
    CLEAR_HOLD_SEC,
    CLICK_DRAG_HOLD_SEC,
    DEADZONE,
    DRAW_PINCH_RATIO,
    DRAW_PINCH_THR_MAX,
    DRAW_PINCH_THR_MIN,
    DRAW_INTERP_STEP_PX,
    DRAW_MIN_POINT_DIST,
    DRAW_OEF_BETA,
    DRAW_OEF_MINCUTOFF,
    DRAW_SMOOTHING,
    EDITOR_COLORS_BGR,
    EDITOR_COLORS_HEX,
    ERASER_SIZE,
    DISPLAY_FPS,
    DISPLAY_EVERY_N,
    FINGER_ANGLE_THRESH,
    FINGER_STATE_METHOD,
    MAX_NUM_HANDS,
    MODE_SWITCH_FRAMES,
    MODE3_ROTATE_GAIN,
    MODE3_SCALE_MAX,
    MODE3_SCALE_MIN,
    MODE3_SEND_FPS,
    MODE3_TRANSLATE_GAIN,
    PINCH_RATIO,
    PINCH_THR_MAX,
    PINCH_THR_MIN,
    PRIMARY_HAND,
    PREDICTION_MAX_NORM_STEP,
    PREDICTION_MS,
    PROC_H,
    PROC_W,
    HAND_SCALE_SMOOTHING,
    LANDMARK_SMOOTHING,
    RELEASE_RATIO,
    RELEASE_THR_MAX,
    RELEASE_THR_MIN,
    RIGHT_CLICK_HOLD_SEC,
    SCROLL_COOLDOWN,
    SCROLL_ENTRY_FRAMES,
    SCROLL_RATIO,
    SCROLL_SENSITIVITY,
    SCROLL_THR_MAX,
    SCROLL_THR_MIN,
    USE_CLAHE,
    VELOCITY_SMOOTHING,
    SWIPE_COOLDOWN,
    SWIPE_MAX_Y,
    SWIPE_MIN_X,
    SWIPE_WINDOW_SEC,
    WINDOW_SWITCH_HOLD_SEC,
    WINDOW_SWITCH_STEP_COOLDOWN,
    WINDOW_SWITCH_STEP_X,
    ZOOM_COOLDOWN,
    ZOOM_RATIO,
    ZOOM_THR_MAX,
    ZOOM_THR_MIN,
    CURSOR_SMOOTH_FRAMES,
)
from .state import (
    draw_queue,
    get_mode,
    screen_h,
    screen_w,
    set_mode,
    toggle_mode,
    try_put,
    try_put_ctrl,
    viewer_queue,
)

from . import settings as _settings


if hasattr(ctypes, "windll"):
    _SET_CURSOR_POS = ctypes.windll.user32.SetCursorPos
else:
    _SET_CURSOR_POS = None


def _move_cursor(x: int, y: int) -> None:
    if _SET_CURSOR_POS is not None:
        _SET_CURSOR_POS(int(x), int(y))
    else:
        pyautogui.moveTo(int(x), int(y))


def _try_put_viewer(msg) -> None:
    try:
        viewer_queue.put_nowait(msg)
    except Exception:
        try:
            viewer_queue.get_nowait()
            viewer_queue.put_nowait(msg)
        except Exception:
            pass


def _get_latest_frame(frame_q):
    frame = frame_q.get()
    while True:
        try:
            frame = frame_q.get_nowait()
        except queue.Empty:
            return frame


def _mode3_measure(h1, h2) -> dict:
    left, right = sorted((h1, h2), key=lambda h: h.wx_n)
    i1 = np.array([left.ix_n, left.iy_n], dtype=np.float32)
    i2 = np.array([right.ix_n, right.iy_n], dtype=np.float32)
    w1 = np.array([left.wx_n, left.wy_n], dtype=np.float32)
    w2 = np.array([right.wx_n, right.wy_n], dtype=np.float32)
    vec = i2 - i1
    mid = (i1 + i2) * 0.5
    wrist_vec = w2 - w1
    return {
        "vec": vec,
        "mid": mid,
        "angle": float(np.arctan2(vec[1], max(abs(vec[0]), 1e-5))),
        "depth": float(left.lm[0].z - right.lm[0].z),
        "wrist_dist": float(max(np.linalg.norm(wrist_vec), 1e-5)),
    }


def _mode3_transform(measure: dict, anchor: dict) -> tuple[float, float, float, float, float, float]:
    mid_delta = measure["mid"] - anchor["mid"]
    vec_delta = measure["vec"] - anchor["vec"]
    raw_yaw = float((mid_delta[0] * 1.4 + vec_delta[0] * 0.8) * MODE3_ROTATE_GAIN)
    raw_pitch = float((mid_delta[1] * 1.2 + vec_delta[1] * 0.8) * MODE3_ROTATE_GAIN)
    raw_roll = float((measure["angle"] - anchor["angle"]) * 1.35 + (measure["depth"] - anchor["depth"]) * 5.0)
    raw_scale = float(np.clip(measure["wrist_dist"] / anchor["wrist_dist"], MODE3_SCALE_MIN, MODE3_SCALE_MAX))
    raw_tx = float(np.clip(mid_delta[0] * MODE3_TRANSLATE_GAIN, -1.4, 1.4))
    raw_ty = float(np.clip(mid_delta[1] * MODE3_TRANSLATE_GAIN, -1.0, 1.0))
    return raw_yaw, raw_pitch, raw_roll, raw_scale, raw_tx, raw_ty


def _predict_point(nx, ny, now, state: dict, key: str):
    if state.get("key") != key or state.get("t") is None:
        state.update({"key": key, "x": nx, "y": ny, "t": now, "vx": 0.0, "vy": 0.0})
        return nx, ny

    dt = max(now - state["t"], 1e-4)
    inst_vx = (nx - state["x"]) / dt
    inst_vy = (ny - state["y"]) / dt
    vx = state["vx"] * VELOCITY_SMOOTHING + inst_vx * (1.0 - VELOCITY_SMOOTHING)
    vy = state["vy"] * VELOCITY_SMOOTHING + inst_vy * (1.0 - VELOCITY_SMOOTHING)
    lead = PREDICTION_MS / 1000.0
    dx = float(np.clip(vx * lead, -PREDICTION_MAX_NORM_STEP, PREDICTION_MAX_NORM_STEP))
    dy = float(np.clip(vy * lead, -PREDICTION_MAX_NORM_STEP, PREDICTION_MAX_NORM_STEP))
    state.update({"key": key, "x": nx, "y": ny, "t": now, "vx": vx, "vy": vy})
    return float(np.clip(nx + dx, 0.0, 1.0)), float(np.clip(ny + dy, 0.0, 1.0))


def _interp_points(x1, y1, x2, y2, step_px):
    dist = float(np.hypot(x2 - x1, y2 - y1))
    if dist <= step_px:
        return [(x2, y2)]
    n = max(2, int(dist / max(step_px, 1)))
    return [
        (int(round(x1 + (x2 - x1) * i / n)), int(round(y1 + (y2 - y1) * i / n)))
        for i in range(1, n + 1)
    ]


_volume_endpoint = None
_volume_unavailable = False
_active_window_unavailable = False


def _set_system_volume_from_y(iy_n: float) -> None:
    global _volume_endpoint, _volume_unavailable
    if _volume_unavailable:
        return
    if _volume_endpoint is None:
        try:
            from ctypes import POINTER, cast

            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            _volume_endpoint = cast(interface, POINTER(IAudioEndpointVolume))
        except Exception:
            _volume_unavailable = True
            return

    vol_db = -65.25 + (1.0 - float(np.clip(iy_n, 0.0, 1.0))) * 65.25
    try:
        _volume_endpoint.SetMasterVolumeLevel(vol_db, None)
    except Exception:
        _volume_unavailable = True


def _load_macros():
    path = pathlib.Path(__file__).resolve().parent / "macros.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return {str(k): list(v) for k, v in data.items() if isinstance(v, list)}
    except Exception:
        return {}


def _fire_macro(macros: dict, name: str) -> None:
    keys = macros.get(name)
    if keys:
        pyautogui.hotkey(*keys)


def _active_profile() -> str:
    global _active_window_unavailable
    if _active_window_unavailable:
        return "default"
    try:
        import pygetwindow as gw

        win = gw.getActiveWindow()
        title = (win.title or "").lower()
        if "photoshop" in title or "paint" in title:
            return "creative"
        if "chrome" in title or "firefox" in title:
            return "browser"
        if "code" in title:
            return "code"
    except Exception:
        _active_window_unavailable = True
        pass
    return "default"


def _reload_settings_globals():
    global _settings
    _settings = importlib.reload(_settings)
    for name in dir(_settings):
        if name.isupper() and name in globals():
            globals()[name] = getattr(_settings, name)


def _hand_key(handedness: str | None, index: int) -> str:
    return f"{(handedness or 'hand').lower()}:{index}"

def _finger_state(lm, tip: int, pip: int, mcp: int) -> bool:
    if FINGER_STATE_METHOD == "angle":
        return is_finger_extended(lm, tip, pip, mcp, angle_deg=FINGER_ANGLE_THRESH)
    return is_finger_up(lm, tip, pip)

@dataclass
class HandState:
    key: str
    handedness: str | None
    lm: list
    hsc: float
    pinch_thr: float
    draw_pinch_thr: float
    release_thr: float
    scroll_thr: float
    zoom_thr: float
    ix_n: float
    iy_n: float
    mx_n: float
    my_n: float
    rx_n: float
    ry_n: float
    px_n: float
    py_n: float
    tx_n: float
    ty_n: float
    wx_n: float
    wy_n: float
    d_click: float
    d_scroll: float
    d_right: float
    d_draw: float
    d_erase: float
    d_color: float
    d_clear: float
    index_up: bool
    middle_up: bool
    index_extended: bool
    middle_extended: bool
    open_palm: bool
    fist: bool
    ily: bool
    fingers_together: bool
    four_fingers_aligned: bool
    gesture_label: str


def _select_primary(hands: list[HandState]) -> HandState | None:
    if not hands:
        return None
    pref = PRIMARY_HAND.lower()
    if pref in ("left", "right"):
        for h in hands:
            if (h.handedness or "").lower() == pref:
                return h
    return max(hands, key=lambda h: h.hsc)


def _select_gesture_hand(hands: list[HandState], primary: HandState | None) -> HandState | None:
    if not hands:
        return None
    if len(hands) == 1 or primary is None:
        return primary
    pref = PRIMARY_HAND.lower()
    target = "left" if pref == "right" else "right"
    for h in hands:
        if (h.handedness or "").lower() == target:
            return h
    for h in hands:
        if h is not primary:
            return h
    return primary


def _preprocess(frame, clahe, clahe_cache, clahe_tick):
    small = cv2.resize(frame, (PROC_W, PROC_H), interpolation=cv2.INTER_NEAREST)
    clahe_tick += 1
    if clahe is not None:
        if clahe_tick % 3 == 0:
            lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            clahe_cache = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        rgb = clahe_cache if clahe_cache is not None else cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    else:
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    return rgb, clahe_cache, clahe_tick


def _draw_skeleton(frame, lm, connections, width, height) -> None:
    for a, b in connections:
        ax = int(lm[a].x * width)
        ay = int(lm[a].y * height)
        bx = int(lm[b].x * width)
        by = int(lm[b].y * height)
        cv2.line(frame, (ax, ay), (bx, by), (80, 80, 220), 1)
    for i, lm_pt in enumerate(lm):
        px = int(lm_pt.x * width)
        py = int(lm_pt.y * height)
        r = 4 if i in (4, 8, 12, 16, 20) else 2
        cv2.circle(frame, (px, py), r, (0, 255, 180), -1)


def _draw_hud(frame, mode, fps, hsc_smooth, pinch_thr, active_profile, has_hand) -> None:
    if DISPLAY_FPS:
        cv2.putText(frame, f"FPS:{fps}", (CAM_W - 70, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
        cv2.putText(frame, f"PROFILE:{active_profile}", (CAM_W - 140, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)

    if has_hand:
        dist_pct = min(1.0, hsc_smooth / 0.55)
        bar_color = (0, 220, 80) if dist_pct > 0.65 else ((0, 200, 255) if dist_pct > 0.35 else (0, 80, 255))
        dist_label = "CLOSE" if dist_pct > 0.65 else ("MEDIUM" if dist_pct > 0.35 else "FAR")
        bar_w = int(dist_pct * 80)
        cv2.rectangle(frame, (CAM_W - 92, 42), (CAM_W - 10, 52), (40, 40, 40), -1)
        cv2.rectangle(frame, (CAM_W - 92, 42), (CAM_W - 92 + bar_w, 52), bar_color, -1)
        cv2.putText(
            frame,
            f"DIST:{dist_label} {pinch_thr:.2f}",
            (CAM_W - 120, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            bar_color,
            1,
        )

    if mode == 1:
        hints = ["idx=move", "fist=scroll", "mid+th=tap/hold", "tap=click hold=drag", "idx+th=R-click", "spread=zoom", "palm=tab"]
    elif mode == 2:
        hints = ["relax=pen up", "idx+th=draw", "mid+th=erase", "rng+th=color", "pky+th(hold)=clear", "S=save"]
    else:
        hints = ["two palms=viewer", "index vector=rotate", "wrist spread=scale", "fist+fist=reset", "ILY+palm=exit"]
    for i, hint in enumerate(hints):
        cv2.putText(frame, hint, (CAM_W - 145, CAM_H - 110 + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)


def _build_hand_state(hand_landmarks, handedness: str | None, key: str, lm_smooth_by_key, hsc_smooth_by_key, lm_alpha: float) -> HandState:
    raw_lm = hand_landmarks.landmark
    raw_coords = np.array([[pt.x, pt.y] for pt in raw_lm], dtype=np.float32)
    
    if key not in lm_smooth_by_key:
        lm_smooth_by_key[key] = raw_coords.copy()
    else:
        # Vectorized exponential averaging: smooth = smooth * alpha + raw * (1 - alpha)
        smooth = lm_smooth_by_key[key]
        np.multiply(smooth, lm_alpha, out=smooth)
        np.multiply(raw_coords, 1 - lm_alpha, out=raw_coords)
        np.add(smooth, raw_coords, out=smooth)

    lm = [LMPoint(float(x), float(y), float(raw_lm[i].z)) for i, (x, y) in enumerate(lm_smooth_by_key[key])]
    raw_hsc = hand_scale(lm)
    prev_hsc = hsc_smooth_by_key.get(key, raw_hsc)
    hsc = prev_hsc * HAND_SCALE_SMOOTHING + raw_hsc * (1 - HAND_SCALE_SMOOTHING)
    hsc_smooth_by_key[key] = hsc

    pinch_thr = float(np.clip(PINCH_RATIO * hsc, PINCH_THR_MIN, PINCH_THR_MAX))
    draw_pinch_thr = float(np.clip(DRAW_PINCH_RATIO * hsc, DRAW_PINCH_THR_MIN, DRAW_PINCH_THR_MAX))
    release_thr = float(np.clip(RELEASE_RATIO * hsc, RELEASE_THR_MIN, RELEASE_THR_MAX))
    scroll_thr = float(np.clip(SCROLL_RATIO * hsc, SCROLL_THR_MIN, SCROLL_THR_MAX))
    zoom_thr = float(np.clip(ZOOM_RATIO * hsc, ZOOM_THR_MIN, ZOOM_THR_MAX))

    ix_n, iy_n = lm[8].x, lm[8].y
    mx_n, my_n = lm[12].x, lm[12].y
    rx_n, ry_n = lm[16].x, lm[16].y
    px_n, py_n = lm[20].x, lm[20].y
    tx_n, ty_n = lm[4].x, lm[4].y
    wx_n, wy_n = lm[0].x, lm[0].y

    # Batch compute all pinch distances (vectorized)
    d_click, d_scroll, d_right, d_draw, d_erase, d_color, d_clear = dist_batch(
        lm, [(12, 4), (8, 12), (12, 4), (8, 4), (12, 4), (16, 4), (20, 4)]
    )

    index_up = _finger_state(lm, 8, 6, 5)
    middle_up = _finger_state(lm, 12, 10, 9)
    index_extended = index_up
    middle_extended = middle_up
    open_palm = sum(1 for tip, pip, mcp in [(8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)] if _finger_state(lm, tip, pip, mcp)) == 4
    fist = is_fist(lm)
    ily = index_up and _finger_state(lm, 20, 18, 17) and not middle_up and not _finger_state(lm, 16, 14, 13)
    fingers_together = are_fingers_together(lm, proximity_threshold=0.10)

    return HandState(
        key=key,
        handedness=handedness,
        lm=lm,
        hsc=hsc,
        pinch_thr=pinch_thr,
        draw_pinch_thr=draw_pinch_thr,
        release_thr=release_thr,
        scroll_thr=scroll_thr,
        zoom_thr=zoom_thr,
        ix_n=ix_n,
        iy_n=iy_n,
        mx_n=mx_n,
        my_n=my_n,
        rx_n=rx_n,
        ry_n=ry_n,
        px_n=px_n,
        py_n=py_n,
        tx_n=tx_n,
        ty_n=ty_n,
        wx_n=wx_n,
        wy_n=wy_n,
        d_click=d_click,
        d_scroll=d_scroll,
        d_right=d_right,
        d_draw=d_draw,
        d_erase=d_erase,
        d_color=d_color,
        d_clear=d_clear,
        index_up=index_up,
        middle_up=middle_up,
        index_extended=index_extended,
        middle_extended=middle_extended,
        open_palm=open_palm,
        fist=fist,
        ily=ily,
        fingers_together=fingers_together,
        four_fingers_aligned=fingers_together,
        gesture_label="",
    )


def webcam_thread(frame_q=None):
    cap = None
    if frame_q is None:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            try_put_ctrl(("quit",))
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        model_complexity=0,
        min_detection_confidence=0.70,
        min_tracking_confidence=0.75,
    )

    lm_alpha = LANDMARK_SMOOTHING

    clahe = None
    if USE_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)

    lm_smooth_by_key = {}
    hsc_smooth_by_key = {}

    hsc_smooth = 0.40
    pinch_thr = PINCH_RATIO * 0.40
    draw_pinch_thr = DRAW_PINCH_RATIO * 0.40
    release_thr = RELEASE_RATIO * 0.40
    scroll_thr = SCROLL_RATIO * 0.40
    zoom_thr = ZOOM_RATIO * 0.40

    # OneEuroFilter for cursor smoothing. Low mincutoff = smooth micro-movements. Lower beta = less speed-based jitter.
    oef_x = OneEuroFilter(freq=60.0, mincutoff=1.5, beta=0.15)
    oef_y = OneEuroFilter(freq=60.0, mincutoff=1.5, beta=0.15)
    cursor_x = screen_w // 2
    cursor_y = screen_h // 2
    last_cursor_x = cursor_x
    last_cursor_y = cursor_y
    cursor_history = deque(maxlen=max(1, CURSOR_SMOOTH_FRAMES))

    gesture_state = "IDLE"
    right_click_start = None
    right_click_done = False
    dragging = False
    click_drag_start = None

    scroll_anchor_y = None
    fingers_together_cnt = 0
    scroll_mode_active = False
    last_scroll_time = 0.0

    zoom_anchor_dist = None
    last_zoom_time = 0.0

    wrist_history = deque()
    last_swipe_time = 0.0
    swipe_flash_label = ""
    swipe_flash_until = 0.0

    window_switch_hold_start = None
    window_switch_active = False
    window_switch_anchor_x = None
    last_window_switch_step = 0.0

    mode_switch_frames = 0
    mode3_frames = 0
    mode3_anchor = None
    mode3_reset_start = None
    mode3_last_send = 0.0
    macros = _load_macros()
    macro_fired = set()
    last_volume_time = 0.0
    last_app_check = 0.0
    active_profile = "default"
    settings_path = pathlib.Path(__file__).resolve().parent / "settings.py"
    settings_mtime = settings_path.stat().st_mtime

    draw_color_idx = 0
    prev_draw_sx = None
    prev_draw_sy = None
    smooth_draw_sx = None
    smooth_draw_sy = None
    draw_oef_x = OneEuroFilter(freq=60.0, mincutoff=DRAW_OEF_MINCUTOFF, beta=DRAW_OEF_BETA)
    draw_oef_y = OneEuroFilter(freq=60.0, mincutoff=DRAW_OEF_MINCUTOFF, beta=DRAW_OEF_BETA)
    drawing_active = False
    stroke_points = []
    stroke_color = EDITOR_COLORS_HEX[0]
    stroke_size_samples = []
    stroke_id_counter = 0
    active_stroke_id = None

    color_change_done = False
    clear_hold_start = None
    clear_done = False
    draw_gesture_active = False
    erase_gesture_active = False
    pointer_predict_state = {"key": None, "x": 0.0, "y": 0.0, "t": None, "vx": 0.0, "vy": 0.0}

    fps_buf = deque(maxlen=30)
    prev_time = time.perf_counter()
    frame_counter = 0

    window_name = "Gesture Controller  |  ESC=quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, CAM_W // 2, CAM_H // 2)
    window_moved = False
    read_failures = 0
    max_read_failures = 20
    
    # OpenCV optimization: use optimized code paths, limit threads to prevent MediaPipe starvation
    cv2.setUseOptimized(True)
    cv2.setNumThreads(2)
    # Low-latency display path: process newest frames and redraw preview every frame by default.
    _clahe_tick = 0
    _disp_tick = 0
    _clahe_cache = None
    
    # Adaptive frame skip is configurable; reusing old MediaPipe results lowers CPU but feels laggy.
    _BUDGET_MS = 18.0  # ~55fps budget for MediaPipe + gesture logic
    _last_result = None
    _skip_next = False
    # Per-stage profiler (toggle with 'P' key)
    from collections import defaultdict as _dd
    _PROF_ON = False
    _prof_buf = _dd(list)
    
    class _T:
        def __init__(self, label):
            self.label = label
            self._t = 0
        def __enter__(self):
            if _PROF_ON:
                self._t = time.perf_counter()
        def __exit__(self, *_):
            if not _PROF_ON: return
            ms = (time.perf_counter() - self._t) * 1000
            b = _prof_buf[self.label]
            b.append(ms)
            if len(b) == 90:
                avg_ms = sum(b) / len(b)
                peak_ms = max(b)
                print(f"[PROF] {self.label:20s}  avg={avg_ms:6.2f}ms  peak={peak_ms:6.2f}ms")
                b.clear()



    def finalize_active_stroke():
        nonlocal drawing_active, stroke_points, stroke_color, stroke_size_samples, active_stroke_id
        nonlocal prev_draw_sx, prev_draw_sy, smooth_draw_sx, smooth_draw_sy

        if drawing_active and active_stroke_id is not None and len(stroke_points) >= 2:
            avg_size = max(BRUSH_MIN, int(np.mean(stroke_size_samples) if stroke_size_samples else BRUSH_MIN))
            try_put(draw_queue, ("stroke_finalize", active_stroke_id, stroke_points, stroke_color, avg_size))

        drawing_active = False
        stroke_points = []
        stroke_size_samples = []
        active_stroke_id = None
        prev_draw_sx = prev_draw_sy = None
        smooth_draw_sx = smooth_draw_sy = None

    while True:
        if frame_q is None:
            success, frame = cap.read()
            if not success:
                read_failures += 1
                if read_failures >= max_read_failures:
                    cap.release()
                    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                    if not cap.isOpened():
                        cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        try_put_ctrl(("quit",))
                        break
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
                    cap.set(cv2.CAP_PROP_FPS, 60)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                    read_failures = 0
                time.sleep(0.005)
                continue
            read_failures = 0
        else:
            frame = _get_latest_frame(frame_q)

        if frame_q is None:
            frame = cv2.flip(frame, 1)
        with _T("preprocess"):
            rgb, _clahe_cache, _clahe_tick = _preprocess(frame, clahe, _clahe_cache, _clahe_tick)

        # Reusing old results is optional; newest real landmarks feel smoother for drawing/cursor control.
        if ADAPTIVE_FRAME_SKIP and _skip_next and _last_result is not None:
            result = _last_result
            _skip_next = False
        else:
            _t0_mp = time.perf_counter()
            with _T("mediapipe"):
                rgb.flags.writeable = False
                try:
                    result = hands.process(rgb)
                finally:
                    rgb.flags.writeable = True
            _last_result = result
            mp_ms = (time.perf_counter() - _t0_mp) * 1000
            if ADAPTIVE_FRAME_SKIP and mp_ms > _BUDGET_MS:
                _skip_next = True

        frame_counter += 1
        if frame_counter % 60 == 0:
            try:
                mtime = settings_path.stat().st_mtime
                if mtime != settings_mtime:
                    _reload_settings_globals()
                    settings_mtime = mtime
                    print("[Settings] Hot-reloaded")
            except OSError:
                pass

        mode = get_mode()

        if mode == 1:
            label, color = "MODE 1: MOUSE", (255, 140, 0)
        elif mode == 2:
            label, color = "MODE 2: SCREEN EDITOR", (0, 210, 100)
        else:
            label, color = "MODE 3: 3D VIEWER", (100, 200, 255)
        cv2.rectangle(frame, (0, 0), (CAM_W, 36), (25, 25, 25), -1)
        cv2.putText(frame, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        now = time.perf_counter()

        if result.multi_hand_landmarks:
            hand_states = []
            handedness_entries = result.multi_handedness or []
            for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                handedness = None
                if idx < len(handedness_entries) and handedness_entries[idx].classification:
                    handedness = handedness_entries[idx].classification[0].label.lower()
                hand_states.append(
                    _build_hand_state(
                        hand_landmarks,
                        handedness,
                        _hand_key(handedness, idx),
                        lm_smooth_by_key,
                        hsc_smooth_by_key,
                        lm_alpha,
                    )
                )

            primary_state = _select_primary(hand_states)
            gesture_hand = _select_gesture_hand(hand_states, primary_state) or primary_state
            if primary_state is None:
                continue

            lm = primary_state.lm
            hsc_smooth = primary_state.hsc
            hsc = hsc_smooth
            pinch_thr = primary_state.pinch_thr
            draw_pinch_thr = primary_state.draw_pinch_thr
            release_thr = primary_state.release_thr
            scroll_thr = primary_state.scroll_thr
            zoom_thr = primary_state.zoom_thr
            ix_n, iy_n = primary_state.ix_n, primary_state.iy_n
            pred_ix_n, pred_iy_n = _predict_point(ix_n, iy_n, time.perf_counter(), pointer_predict_state, primary_state.key)
            mx_n, my_n = primary_state.mx_n, primary_state.my_n
            rx_n, ry_n = primary_state.rx_n, primary_state.ry_n
            px_n, py_n = primary_state.px_n, primary_state.py_n
            tx_n, ty_n = primary_state.tx_n, primary_state.ty_n
            wx_n, wy_n = primary_state.wx_n, primary_state.wy_n

            d_click = primary_state.d_click
            d_scroll = primary_state.d_scroll
            d_right = primary_state.d_right
            d_draw = primary_state.d_draw
            d_erase = primary_state.d_erase
            d_color = primary_state.d_color
            d_clear = primary_state.d_clear

            index_up = primary_state.index_up
            middle_up = primary_state.middle_up
            fist = primary_state.fist
            gesture_lm = gesture_hand.lm
            gesture_ix_n, gesture_iy_n = gesture_hand.ix_n, gesture_hand.iy_n
            gesture_mx_n, gesture_my_n = gesture_hand.mx_n, gesture_hand.my_n
            gesture_rx_n, gesture_ry_n = gesture_hand.rx_n, gesture_hand.ry_n
            gesture_px_n, gesture_py_n = gesture_hand.px_n, gesture_hand.py_n
            gesture_tx_n, gesture_ty_n = gesture_hand.tx_n, gesture_hand.ty_n
            gesture_wx_n, gesture_wy_n = gesture_hand.wx_n, gesture_hand.wy_n
            gesture_d_click = gesture_hand.d_click
            gesture_d_scroll = gesture_hand.d_scroll
            gesture_d_right = gesture_hand.d_right
            gesture_d_draw = gesture_hand.d_draw
            gesture_d_erase = gesture_hand.d_erase
            gesture_d_color = gesture_hand.d_color
            gesture_d_clear = gesture_hand.d_clear
            gesture_pinch_thr = gesture_hand.pinch_thr
            gesture_draw_pinch_thr = gesture_hand.draw_pinch_thr
            gesture_release_thr = gesture_hand.release_thr
            gesture_scroll_thr = gesture_hand.scroll_thr
            gesture_zoom_thr = gesture_hand.zoom_thr
            gesture_index_up = gesture_hand.index_up
            gesture_middle_up = gesture_hand.middle_up
            gesture_open_palm = gesture_hand.open_palm

            _draw_skeleton(frame, primary_state.lm, mp_hands.HAND_CONNECTIONS, CAM_W, CAM_H)

            # OneEuroFilter cursor smoothing: adaptive, responsive, low-latency
            raw_sx, raw_sy = norm_to_screen(pred_ix_n, pred_iy_n, screen_w, screen_h)
            t_now = time.perf_counter()
            
            # Apply OneEuroFilter
            filtered_x = oef_x.filter(float(raw_sx), t_now)
            filtered_y = oef_y.filter(float(raw_sy), t_now)
            
            # Apply median/moving avg filtering for extra smoothness against spikes
            cursor_history.append((filtered_x, filtered_y))
            hist_x = [pt[0] for pt in cursor_history]
            hist_y = [pt[1] for pt in cursor_history]
            avg_x = sorted(hist_x)[len(hist_x) // 2] if len(hist_x) >= 3 else sum(hist_x) / len(hist_x)
            avg_y = sorted(hist_y)[len(hist_y) // 2] if len(hist_y) >= 3 else sum(hist_y) / len(hist_y)
            
            cursor_x = int(avg_x)
            cursor_y = int(avg_y)

            if len(hand_states) == 2:
                h1, h2 = hand_states[0], hand_states[1]
                if h1.open_palm and h2.open_palm:
                    mode3_frames += 1
                    if mode3_frames >= MODE_SWITCH_FRAMES and mode == 1:
                        mode3_anchor = _mode3_measure(h1, h2)
                        set_mode(3)
                        mode = 3
                        mode3_frames = 0
                        try_put_ctrl(("mode", 3))
                        _try_put_viewer(("active", True))
                        _try_put_viewer(("reset",))
                else:
                    mode3_frames = 0

                if mode == 3:
                    measure = _mode3_measure(h1, h2)
                    if mode3_anchor is None:
                        mode3_anchor = measure
                        _try_put_viewer(("active", True))
                    if now - mode3_last_send >= 1.0 / max(MODE3_SEND_FPS, 1):
                        raw_yaw, raw_pitch, raw_roll, raw_scale, raw_tx, raw_ty = _mode3_transform(measure, mode3_anchor)
                        _try_put_viewer(("transform", raw_yaw, raw_pitch, raw_roll, raw_scale, raw_tx, raw_ty))
                        mode3_last_send = now

                    if h1.fist and h2.fist:
                        if mode3_reset_start is None:
                            mode3_reset_start = now
                        elif now - mode3_reset_start >= 1.5:
                            mode3_anchor = measure
                            _try_put_viewer(("reset",))
                            mode3_reset_start = now
                    else:
                        mode3_reset_start = None

                    if (h1.ily and h2.open_palm) or (h2.ily and h1.open_palm):
                        set_mode(1)
                        mode = 1
                        mode3_anchor = None
                        try_put_ctrl(("mode", 1))
                        _try_put_viewer(("active", False))
            else:
                mode3_frames = 0
                mode3_reset_start = None

            if mode != 3 and is_ily_gesture(gesture_lm):
                mode_switch_frames += 1
                prog = int((mode_switch_frames / MODE_SWITCH_FRAMES) * (CAM_W - 20))
                cv2.rectangle(frame, (10, CAM_H - 22), (10 + prog, CAM_H - 10), (0, 255, 200), -1)
                cv2.putText(
                    frame,
                    "\\m/ Hold to switch mode...",
                    (10, CAM_H - 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (0, 255, 200),
                    1,
                )
                if mode_switch_frames >= MODE_SWITCH_FRAMES:
                    if "ILY_HOLD" not in macro_fired:
                        _fire_macro(macros, "ILY_HOLD")
                        macro_fired.add("ILY_HOLD")
                    if window_switch_active:
                        pyautogui.keyUp("alt")
                        window_switch_active = False
                    new_mode = toggle_mode()
                    mode_switch_frames = 0
                    gesture_state = "IDLE"
                    right_click_start = None
                    right_click_done = False
                    scroll_mode_active = False
                    scroll_anchor_y = None
                    zoom_anchor_dist = None
                    finalize_active_stroke()
                    clear_hold_start = None
                    clear_done = False
                    draw_gesture_active = False
                    erase_gesture_active = False
                    window_switch_hold_start = None
                    window_switch_anchor_x = None
                    try_put_ctrl(("mode", new_mode))
                    if new_mode != 3:
                        mode3_anchor = None
                        _try_put_viewer(("active", False))
            else:
                mode_switch_frames = 0
                macro_fired.discard("ILY_HOLD")

            if mode == 1:
                try_put_ctrl(("cursor_hide",))

                if now - last_app_check > 2.0:
                    last_app_check = now
                    active_profile = _active_profile()

                if len(hand_states) > 1 and gesture_hand is not primary_state and gesture_hand.ily and now - last_volume_time > 0.08:
                    _set_system_volume_from_y(gesture_iy_n)
                    last_volume_time = now
                    cv2.putText(frame, "VOLUME", (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 210, 255), 2)

                window_switch_pinched = gesture_d_color < (gesture_release_thr if window_switch_active else gesture_pinch_thr)

                if window_switch_pinched:
                    if not window_switch_active:
                        if window_switch_hold_start is None:
                            window_switch_hold_start = now

                        hold_pct = min(1.0, (now - window_switch_hold_start) / WINDOW_SWITCH_HOLD_SEC)
                        rx_px = int(gesture_rx_n * CAM_W)
                        ry_px = int(gesture_ry_n * CAM_H)
                        cv2.ellipse(
                            frame,
                            (rx_px, ry_px),
                            (18, 18),
                            -90,
                            0,
                            int(360 * hold_pct),
                            (255, 120, 0),
                            2,
                        )
                        cv2.putText(
                            frame,
                            "HOLD: WINDOW SWITCH",
                            (10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.58,
                            (255, 120, 0),
                            2,
                        )

                        if hold_pct >= 1.0:
                            pyautogui.keyDown("alt")
                            pyautogui.press("tab")
                            window_switch_active = True
                            window_switch_anchor_x = (gesture_tx_n + gesture_rx_n) / 2.0
                            last_window_switch_step = now
                            scroll_mode_active = False
                            scroll_anchor_y = None
                            zoom_anchor_dist = None
                    else:
                        pinch_center_x = (gesture_tx_n + gesture_rx_n) / 2.0
                        if window_switch_anchor_x is None:
                            window_switch_anchor_x = pinch_center_x

                        delta_x = pinch_center_x - window_switch_anchor_x
                        if abs(delta_x) > WINDOW_SWITCH_STEP_X and (now - last_window_switch_step) > WINDOW_SWITCH_STEP_COOLDOWN:
                            if delta_x > 0:
                                pyautogui.press("tab")
                            else:
                                pyautogui.hotkey("shift", "tab")
                            window_switch_anchor_x = pinch_center_x
                            last_window_switch_step = now

                        cv2.putText(
                            frame,
                            "WINDOW SWITCH (MOVE PINCH)",
                            (10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.58,
                            (255, 200, 0),
                            2,
                        )
                else:
                    window_switch_hold_start = None
                    window_switch_anchor_x = None
                    if window_switch_active:
                        pyautogui.keyUp("alt")
                        window_switch_active = False

                wrist_history.append((gesture_wx_n, gesture_wy_n, now))
                while wrist_history and now - wrist_history[0][2] > SWIPE_WINDOW_SEC:
                    wrist_history.popleft()

                if is_open_palm(gesture_lm) and len(wrist_history) >= 4 and (now - last_swipe_time) > SWIPE_COOLDOWN:
                    x0, y0, _ = wrist_history[0]
                    net_x = gesture_wx_n - x0
                    net_y = abs(gesture_wy_n - y0)
                    if abs(net_x) > SWIPE_MIN_X and net_y < SWIPE_MAX_Y:
                        if net_x > 0:
                            pyautogui.hotkey("ctrl", "shift", "tab")
                            swipe_flash_label = "<< PREV TAB"
                        else:
                            pyautogui.hotkey("ctrl", "tab")
                            swipe_flash_label = "NEXT TAB >>"
                        last_swipe_time = now
                        swipe_flash_until = now + 0.8
                        wrist_history.clear()

                if now < swipe_flash_until:
                    cv2.putText(
                        frame,
                        swipe_flash_label,
                        (CAM_W // 2 - 80, 66),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 220, 255),
                        2,
                    )

                if d_right < pinch_thr and not index_up and not scroll_mode_active:
                    if right_click_start is None:
                        right_click_start = now
                    elif not right_click_done and now - right_click_start > RIGHT_CLICK_HOLD_SEC:
                        pyautogui.rightClick()
                        right_click_done = True
                    pct = min(1.0, (now - right_click_start) / RIGHT_CLICK_HOLD_SEC)
                    tx_px = int(tx_n * CAM_W)
                    ty_px = int(ty_n * CAM_H)
                    cv2.ellipse(frame, (tx_px, ty_px), (18, 18), -90, 0, int(360 * pct), (0, 100, 255), 3)
                    cv2.putText(frame, "RIGHT CLICK", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
                else:
                    right_click_start = None
                    right_click_done = False

                ring_dn = not is_finger_up(lm, 16, 14)
                pinky_dn = not is_finger_up(lm, 20, 18)
                zoom_norm = dist_norm(gesture_lm, 8, 12)
                if gesture_index_up and gesture_middle_up and ring_dn and pinky_dn and zoom_norm > gesture_zoom_thr:
                    if zoom_anchor_dist is None:
                        zoom_anchor_dist = zoom_norm
                    else:
                        delta = zoom_norm - zoom_anchor_dist
                        if abs(delta) > 0.018 and (now - last_zoom_time) > ZOOM_COOLDOWN:
                            pyautogui.hotkey("ctrl", "+" if delta > 0 else "-")
                            zoom_anchor_dist = zoom_norm
                            last_zoom_time = now
                    ix_px = int(gesture_ix_n * CAM_W)
                    iy_px = int(gesture_iy_n * CAM_H)
                    mx_px = int(gesture_mx_n * CAM_W)
                    my_px = int(gesture_my_n * CAM_H)
                    cv2.line(frame, (ix_px, iy_px), (mx_px, my_px), (255, 200, 0), 2)
                    cv2.putText(frame, "ZOOM", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
                else:
                    zoom_anchor_dist = None

                # Scroll feature: activate when hand is closed in a fist (like grabbing the page)
                if fist:
                    fingers_together_cnt += 1
                    if fingers_together_cnt >= SCROLL_ENTRY_FRAMES:
                        scroll_mode_active = True

                    if scroll_mode_active:
                        if scroll_anchor_y is None:
                            # Use wrist Y position for scroll reference since fingers are curled
                            scroll_anchor_y = wy_n
                        
                        scroll_delta = wy_n - scroll_anchor_y
                        
                        if abs(scroll_delta) > 0.005 and (now - last_scroll_time) > SCROLL_COOLDOWN:
                            # Hand moving down (delta > 0) = scroll down (direction = -1 on Windows)
                            direction = -1 if scroll_delta > 0 else 1
                            # Exaggerate delta slightly for smoother scrolling
                            scroll_ticks = max(1, int(abs(scroll_delta) * 100))
                            pyautogui.scroll(int(direction * scroll_ticks * SCROLL_SENSITIVITY))
                            last_scroll_time = now
                            # Smoothly follow anchor rather than hard reset to reduce jitter
                            scroll_anchor_y = scroll_anchor_y + (scroll_delta * 0.8)
                        
                        anc_py = int(scroll_anchor_y * CAM_H)
                        cv2.line(frame, (0, anc_py), (CAM_W, anc_py), (0, 255, 255), 1)
                        
                        wx_px = int(wx_n * CAM_W)
                        wy_px = int(wy_n * CAM_H)
                        cv2.circle(frame, (wx_px, wy_px), 12, (0, 255, 255), -1)
                        cv2.putText(frame, "SCROLLING", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                else:
                    fingers_together_cnt = 0
                    scroll_mode_active = False
                    scroll_anchor_y = None

                    if zoom_anchor_dist is None:
                        moved_x = abs(cursor_x - last_cursor_x)
                        moved_y = abs(cursor_y - last_cursor_y)
                        if moved_x > DEADZONE or moved_y > DEADZONE:
                            # Cap sudden jumps from transient landmark glitches.
                            dx = cursor_x - last_cursor_x
                            dy = cursor_y - last_cursor_y
                            step_dist = float(np.hypot(dx, dy))
                            if step_dist > 180.0:
                                scale = 180.0 / step_dist
                                move_x = int(last_cursor_x + dx * scale)
                                move_y = int(last_cursor_y + dy * scale)
                            else:
                                move_x = cursor_x
                                move_y = cursor_y

                            with _T("mouse"):
                                _move_cursor(move_x, move_y)
                            last_cursor_x = move_x
                            last_cursor_y = move_y

                        ix_px = int(ix_n * CAM_W)
                        iy_px = int(iy_n * CAM_H)
                        cv2.circle(frame, (ix_px, iy_px), 8, (255, 140, 0), 2)
                        cv2.putText(frame, "MOVING", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 140, 0), 2)

                click_drag_active = (d_click < (release_thr if dragging else pinch_thr)) and not scroll_mode_active
                if click_drag_active:
                    if click_drag_start is None:
                        click_drag_start = now

                    hold_t = now - click_drag_start
                    if hold_t >= CLICK_DRAG_HOLD_SEC:
                        if not dragging:
                            pyautogui.mouseDown()
                            dragging = True
                        mx_px = int(mx_n * CAM_W)
                        my_px = int(my_n * CAM_H)
                        cv2.circle(frame, (mx_px, my_px), 14, (0, 0, 255), 3)
                        cv2.putText(frame, "DRAG (HOLD)", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        pct = min(1.0, hold_t / CLICK_DRAG_HOLD_SEC)
                        mx_px = int(mx_n * CAM_W)
                        my_px = int(my_n * CAM_H)
                        cv2.ellipse(frame, (mx_px, my_px), (16, 16), -90, 0, int(360 * pct), (0, 255, 120), 2)
                        cv2.putText(frame, "CLICK READY", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2)
                else:
                    if click_drag_start is not None:
                        hold_t = now - click_drag_start
                        if hold_t < CLICK_DRAG_HOLD_SEC and not dragging:
                            pyautogui.click()
                            gesture_state = "CLICKED"
                        click_drag_start = None

                    if dragging:
                        pyautogui.mouseUp()
                        dragging = False

                    if gesture_state == "CLICKED" and d_click > release_thr:
                        gesture_state = "IDLE"

                cv2.putText(
                    frame,
                    f"Click:{gesture_state}  Drag:{dragging}",
                    (10, 185),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (160, 160, 160),
                    1,
                )

            elif mode == 2:
                cur_hex = EDITOR_COLORS_HEX[draw_color_idx]
                cur_bgr = EDITOR_COLORS_BGR[draw_color_idx]

                sx, sy = norm_to_screen(ix_n, iy_n, screen_w, screen_h)

                if d_clear < pinch_thr:
                    finalize_active_stroke()
                    draw_gesture_active = False
                    erase_gesture_active = False
                    if clear_hold_start is None:
                        clear_hold_start = now
                    pct = min(1.0, (now - clear_hold_start) / CLEAR_HOLD_SEC)
                    px_px = int(px_n * CAM_W)
                    py_px = int(py_n * CAM_H)
                    cv2.ellipse(frame, (px_px, py_px), (22, 22), -90, 0, int(360 * pct), (0, 80, 255), 3)
                    cv2.putText(frame, "Hold to clear...", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2)
                    if pct >= 1.0 and not clear_done:
                        try_put(draw_queue, ("clear",))
                        prev_draw_sx = prev_draw_sy = None
                        clear_done = True
                    try_put_ctrl(("cursor", sx, sy, "#555555", "penup"))

                else:
                    clear_hold_start = None
                    clear_done = False

                    if d_color < pinch_thr:
                        finalize_active_stroke()
                        draw_gesture_active = False
                        erase_gesture_active = False
                        if not color_change_done:
                            draw_color_idx = (draw_color_idx + 1) % len(EDITOR_COLORS_HEX)
                            color_change_done = True
                            try_put(draw_queue, ("color", draw_color_idx))
                        rx_px = int(rx_n * CAM_W)
                        ry_px = int(ry_n * CAM_H)
                        cv2.circle(frame, (rx_px, ry_px), 14, EDITOR_COLORS_BGR[draw_color_idx], 3)
                        cv2.putText(
                            frame,
                            "COLOR",
                            (10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            EDITOR_COLORS_BGR[draw_color_idx],
                            2,
                        )
                        try_put_ctrl(("cursor", sx, sy, EDITOR_COLORS_HEX[draw_color_idx], "penup"))
                        prev_draw_sx = prev_draw_sy = None

                    else:
                        color_change_done = False

                        erase_gesture_active = d_erase < (release_thr if erase_gesture_active else draw_pinch_thr)
                        draw_gesture_active = d_draw < (release_thr if draw_gesture_active else draw_pinch_thr)

                        if erase_gesture_active:
                            finalize_active_stroke()
                            draw_gesture_active = False
                            esx, esy = norm_to_screen((mx_n + tx_n) / 2, (my_n + ty_n) / 2, screen_w, screen_h)
                            try_put(draw_queue, ("erase", esx, esy, ERASER_SIZE * 2))
                            try_put_ctrl(("cursor", esx, esy, "#aaaaaa", "erase"))
                            mx_px = int(mx_n * CAM_W)
                            my_px = int(my_n * CAM_H)
                            cv2.circle(frame, (mx_px, my_px), 14, (180, 180, 180), 2)
                            cv2.putText(frame, "ERASING", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

                        elif draw_gesture_active:
                            erase_gesture_active = False
                            spread_pct = min(1.0, d_draw / draw_pinch_thr)
                            brush_size = int(BRUSH_MIN + spread_pct * (BRUSH_MAX - BRUSH_MIN))

                            track_x = ix_n
                            track_y = iy_n

                            if smooth_draw_sx is None:
                                smooth_draw_sx = float(track_x)
                                smooth_draw_sy = float(track_y)
                            else:
                                smooth_draw_sx = smooth_draw_sx * DRAW_SMOOTHING + track_x * (1 - DRAW_SMOOTHING)
                                smooth_draw_sy = smooth_draw_sy * DRAW_SMOOTHING + track_y * (1 - DRAW_SMOOTHING)

                            raw_dsx, raw_dsy = norm_to_screen(smooth_draw_sx, smooth_draw_sy, screen_w, screen_h)
                            t_draw = time.perf_counter()
                            dsx = int(draw_oef_x.filter(float(raw_dsx), t_draw))
                            dsy = int(draw_oef_y.filter(float(raw_dsy), t_draw))

                            if not drawing_active:
                                stroke_id_counter += 1
                                active_stroke_id = stroke_id_counter
                                drawing_active = True
                                stroke_points = []
                                stroke_size_samples = []
                                stroke_color = cur_hex
                                try_put(draw_queue, ("stroke_begin", active_stroke_id))

                            if not stroke_points or np.hypot(stroke_points[-1][0] - dsx, stroke_points[-1][1] - dsy) >= DRAW_MIN_POINT_DIST:
                                stroke_points.append((dsx, dsy))
                                stroke_size_samples.append(brush_size)

                            if prev_draw_sx is not None and active_stroke_id is not None:
                                interp = _interp_points(prev_draw_sx, prev_draw_sy, dsx, dsy, DRAW_INTERP_STEP_PX)
                                batch_points = [(prev_draw_sx, prev_draw_sy), *interp]
                                try_put(draw_queue, ("draw_batch", active_stroke_id, batch_points, cur_hex, brush_size))
                                prev_draw_sx, prev_draw_sy = batch_points[-1]

                            if prev_draw_sx is None:
                                prev_draw_sx, prev_draw_sy = dsx, dsy
                            try_put_ctrl(("cursor", dsx, dsy, cur_hex, "draw"))

                            d_px = int(track_x * CAM_W)
                            d_py = int(track_y * CAM_H)
                            cv2.circle(frame, (d_px, d_py), brush_size + 2, cur_bgr, -1)
                            cv2.putText(
                                frame,
                                f"DRAWING  brush:{brush_size}",
                                (10, 160),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                cur_bgr,
                                2,
                            )

                        else:
                            draw_gesture_active = False
                            erase_gesture_active = False
                            finalize_active_stroke()
                            try_put_ctrl(("cursor", sx, sy, "#777777", "penup"))
                            cv2.putText(frame, "PEN UP", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (130, 130, 130), 2)

                cv2.rectangle(frame, (CAM_W - 40, 44), (CAM_W - 8, 76), cur_bgr, -1)
                cv2.rectangle(frame, (CAM_W - 40, 44), (CAM_W - 8, 76), (255, 255, 255), 1)

        else:
            if window_switch_active:
                pyautogui.keyUp("alt")
                window_switch_active = False
            lm_smooth = None
            scroll_mode_active = False
            scroll_anchor_y = None
            zoom_anchor_dist = None
            right_click_start = None
            right_click_done = False
            click_drag_start = None
            clear_hold_start = None
            window_switch_hold_start = None
            window_switch_anchor_x = None
            draw_gesture_active = False
            erase_gesture_active = False
            finalize_active_stroke()
            if dragging:
                pyautogui.mouseUp()
                dragging = False
            try_put_ctrl(("cursor_hide",))
            cv2.putText(
                frame,
                "NO HAND DETECTED",
                (CAM_W // 2 - 110, CAM_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (60, 60, 255),
                2,
            )

        fps_buf.append(1 / max(now - prev_time, 0.001))
        prev_time = now
        fps = int(sum(fps_buf) / len(fps_buf))
        _draw_hud(frame, get_mode(), fps, hsc_smooth, pinch_thr, active_profile, bool(result.multi_hand_landmarks))

        # Redraw preview every frame by default so landmarks track fast hand motion.
        _disp_tick += 1
        if _disp_tick % max(DISPLAY_EVERY_N, 1) == 0:
            with _T("imshow"):
                cv2.imshow(window_name, frame)

        if not window_moved:
            cv2.moveWindow(window_name, CAM_WIN_X, CAM_WIN_Y)
            window_moved = True

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            if window_switch_active:
                pyautogui.keyUp("alt")
                window_switch_active = False
            try_put_ctrl(("quit",))
            break
        if key == ord("s"):
            try_put_ctrl(("save",))
        elif key == ord("c"):
            try_put_ctrl(("clear",))
        elif key == ord("p"):
            _PROF_ON = not _PROF_ON
            print(f"[PROFILER] {'ON' if _PROF_ON else 'OFF'}")
    
    if window_switch_active:
        pyautogui.keyUp("alt")
        window_switch_active = False

    if cap is not None:
        cap.release()
    hands.close()
    cv2.destroyAllWindows()

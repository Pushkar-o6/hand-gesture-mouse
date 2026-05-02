import ctypes
import pathlib
import time
from collections import deque
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

from .helpers import (
    LMPoint,
    dist_batch,
    dist_norm,
    hand_scale,
    is_finger_up,
    is_finger_extended,
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
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID,
    CLEAR_HOLD_SEC,
    CLICK_DRAG_HOLD_SEC,
    CURSOR_SMOOTH_FRAMES,
    DEADZONE,
    DRAW_PINCH_RATIO,
    DRAW_PINCH_THR_MAX,
    DRAW_PINCH_THR_MIN,
    DRAW_SMOOTHING,
    EDITOR_COLORS_BGR,
    EDITOR_COLORS_HEX,
    ERASER_SIZE,
    DISPLAY_FPS,
    DISPLAY_VISIBILITY_POLL_SEC,
    FINGER_ANGLE_THRESH,
    FINGER_STATE_METHOD,
    GESTURE_MODEL_PATH,
    MAX_NUM_HANDS,
    MODEL_MODE_LABELS,
    MODEL_SCROLL_LABELS,
    MODEL_ZOOM_LABELS,
    MODE_SWITCH_FRAMES,
    PINCH_RATIO,
    PINCH_THR_MAX,
    PINCH_THR_MIN,
    PRIMARY_HAND,
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
    USE_GESTURE_MODEL,
    SMOOTHING,
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
)
from .state import (
    draw_queue,
    get_mode,
    screen_h,
    screen_w,
    toggle_mode,
    try_put,
    try_put_ctrl,
)

from .gesture_ml import GestureKNN, extract_features


if hasattr(ctypes, "windll"):
    _SET_CURSOR_POS = ctypes.windll.user32.SetCursorPos
else:
    _SET_CURSOR_POS = None


def _move_cursor(x: int, y: int) -> None:
    if _SET_CURSOR_POS is not None:
        _SET_CURSOR_POS(int(x), int(y))
    else:
        pyautogui.moveTo(int(x), int(y))


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
    ily: bool
    gesture_label: str


def _resolve_model_path(path_str: str) -> pathlib.Path:
    path = pathlib.Path(path_str)
    if path.is_absolute():
        return path
    return pathlib.Path(__file__).resolve().parent / path


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

    lm = [LMPoint(float(x), float(y)) for x, y in lm_smooth_by_key[key]]
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
    ily = index_up and _finger_state(lm, 20, 18, 17) and not middle_up and not _finger_state(lm, 16, 14, 13)

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
        ily=ily,
        gesture_label="",
    )


def webcam_thread():
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
        min_detection_confidence=0.60,
        min_tracking_confidence=0.60,
    )

    lm_alpha = LANDMARK_SMOOTHING

    gesture_model = None
    if USE_GESTURE_MODEL:
        model_path = _resolve_model_path(GESTURE_MODEL_PATH)
        gesture_model = GestureKNN.load(str(model_path))

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

    cx_buf = deque(maxlen=CURSOR_SMOOTH_FRAMES)
    cy_buf = deque(maxlen=CURSOR_SMOOTH_FRAMES)
    cursor_x = screen_w // 2
    cursor_y = screen_h // 2
    last_cursor_x = cursor_x
    last_cursor_y = cursor_y
    last_target_x = cursor_x
    last_target_y = cursor_y

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

    draw_color_idx = 0
    prev_draw_sx = None
    prev_draw_sy = None
    smooth_draw_sx = None
    smooth_draw_sy = None
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

    fps_buf = deque(maxlen=30)
    prev_time = time.time()

    window_name = "Gesture Controller  |  ESC=quit"
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
            try_put(draw_queue, ("stroke_finalize", active_stroke_id, stroke_points, stroke_color, avg_size))

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

        frame = cv2.flip(frame, 1)
        small = cv2.resize(frame, (PROC_W, PROC_H))
        if clahe is not None:
            lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        mode = get_mode()

        label = "MODE 1: MOUSE" if mode == 1 else "MODE 2: SCREEN EDITOR"
        color = (255, 140, 0) if mode == 1 else (0, 210, 100)
        cv2.rectangle(frame, (0, 0), (CAM_W, 36), (25, 25, 25), -1)
        cv2.putText(frame, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        now = time.time()

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
            gesture_state = _select_gesture_hand(hand_states, primary_state) or primary_state
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
            gesture_lm = gesture_state.lm
            gesture_ix_n, gesture_iy_n = gesture_state.ix_n, gesture_state.iy_n
            gesture_mx_n, gesture_my_n = gesture_state.mx_n, gesture_state.my_n
            gesture_rx_n, gesture_ry_n = gesture_state.rx_n, gesture_state.ry_n
            gesture_px_n, gesture_py_n = gesture_state.px_n, gesture_state.py_n
            gesture_tx_n, gesture_ty_n = gesture_state.tx_n, gesture_state.ty_n
            gesture_wx_n, gesture_wy_n = gesture_state.wx_n, gesture_state.wy_n
            gesture_d_click = gesture_state.d_click
            gesture_d_scroll = gesture_state.d_scroll
            gesture_d_right = gesture_state.d_right
            gesture_d_draw = gesture_state.d_draw
            gesture_d_erase = gesture_state.d_erase
            gesture_d_color = gesture_state.d_color
            gesture_d_clear = gesture_state.d_clear
            gesture_pinch_thr = gesture_state.pinch_thr
            gesture_draw_pinch_thr = gesture_state.draw_pinch_thr
            gesture_release_thr = gesture_state.release_thr
            gesture_scroll_thr = gesture_state.scroll_thr
            gesture_zoom_thr = gesture_state.zoom_thr
            gesture_index_up = gesture_state.index_up
            gesture_middle_up = gesture_state.middle_up
            gesture_open_palm = gesture_state.open_palm

            for conn in mp_hands.HAND_CONNECTIONS:
                a, b = conn
                ax = int(primary_state.lm[a].x * CAM_W)
                ay = int(primary_state.lm[a].y * CAM_H)
                bx = int(primary_state.lm[b].x * CAM_W)
                by = int(primary_state.lm[b].y * CAM_H)
                cv2.line(frame, (ax, ay), (bx, by), (80, 80, 220), 1)
            for i, lm_pt in enumerate(primary_state.lm):
                px = int(lm_pt.x * CAM_W)
                py = int(lm_pt.y * CAM_H)
                r = 4 if i in (4, 8, 12, 16, 20) else 2
                cv2.circle(frame, (px, py), r, (0, 255, 180), -1)

            tx_s, ty_s = norm_to_screen(ix_n, iy_n, screen_w, screen_h)
            cx_buf.append(tx_s)
            cy_buf.append(ty_s)
            cx_med = int(sorted(cx_buf)[len(cx_buf) // 2])
            cy_med = int(sorted(cy_buf)[len(cy_buf) // 2])

            # Blend median and raw target to keep stability without adding too much latency.
            target_x = int(cx_med * 0.70 + tx_s * 0.30)
            target_y = int(cy_med * 0.70 + ty_s * 0.30)

            # Adaptive smoothing: smooth more when hand is steady, less when moving fast.
            speed_px = float(np.hypot(target_x - last_target_x, target_y - last_target_y))
            speed_norm = float(np.clip((speed_px - 4.0) / 80.0, 0.0, 1.0))
            adaptive_smoothing = max(0.08, SMOOTHING - 0.22 * speed_norm)

            # Tiny prediction improves responsiveness for quick cursor motion.
            pred_x = int(target_x + 0.15 * (target_x - last_target_x))
            pred_y = int(target_y + 0.15 * (target_y - last_target_y))
            pred_x = int(np.clip(pred_x, 0, screen_w - 1))
            pred_y = int(np.clip(pred_y, 0, screen_h - 1))

            cursor_x = int(cursor_x * adaptive_smoothing + pred_x * (1 - adaptive_smoothing))
            cursor_y = int(cursor_y * adaptive_smoothing + pred_y * (1 - adaptive_smoothing))

            last_target_x = target_x
            last_target_y = target_y

            if is_ily_gesture(gesture_lm):
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
                    cx_buf.clear()
                    cy_buf.clear()
                    try_put_ctrl(("mode", new_mode))
            else:
                mode_switch_frames = 0

            if mode == 1:
                try_put(draw_queue, ("cursor_hide",))

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

                if gesture_d_scroll < gesture_scroll_thr:
                    fingers_together_cnt += 1
                    if fingers_together_cnt >= SCROLL_ENTRY_FRAMES:
                        scroll_mode_active = True

                    if scroll_mode_active:
                        if scroll_anchor_y is None:
                            scroll_anchor_y = gesture_iy_n
                        scroll_delta = gesture_iy_n - scroll_anchor_y
                        if abs(scroll_delta) > 0.005 and (now - last_scroll_time) > SCROLL_COOLDOWN:
                            direction = -1 if scroll_delta > 0 else 1
                            scroll_ticks = max(1, int(abs(scroll_delta) * 40))
                            pyautogui.scroll(direction * scroll_ticks * SCROLL_SENSITIVITY)
                            last_scroll_time = now
                        anc_py = int(scroll_anchor_y * CAM_H)
                        cv2.line(frame, (0, anc_py), (CAM_W, anc_py), (0, 255, 255), 1)
                        ix_px = int(gesture_ix_n * CAM_W)
                        iy_px = int(gesture_iy_n * CAM_H)
                        cv2.circle(frame, (ix_px, iy_px), 10, (0, 255, 255), 2)
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
                    try_put(draw_queue, ("cursor", sx, sy, "#555555", "penup"))

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
                        try_put(draw_queue, ("cursor", sx, sy, EDITOR_COLORS_HEX[draw_color_idx], "penup"))
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
                            try_put(draw_queue, ("cursor", esx, esy, "#aaaaaa", "erase"))
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

                            dsx, dsy = norm_to_screen(smooth_draw_sx, smooth_draw_sy, screen_w, screen_h)

                            if not drawing_active:
                                stroke_id_counter += 1
                                active_stroke_id = stroke_id_counter
                                drawing_active = True
                                stroke_points = []
                                stroke_size_samples = []
                                stroke_color = cur_hex
                                try_put(draw_queue, ("stroke_begin", active_stroke_id))

                            if not stroke_points or abs(stroke_points[-1][0] - dsx) + abs(stroke_points[-1][1] - dsy) >= 2:
                                stroke_points.append((dsx, dsy))
                                stroke_size_samples.append(brush_size)

                            if prev_draw_sx is not None and active_stroke_id is not None:
                                try_put(
                                    draw_queue,
                                    ("draw", prev_draw_sx, prev_draw_sy, dsx, dsy, cur_hex, brush_size, active_stroke_id),
                                )

                            prev_draw_sx, prev_draw_sy = dsx, dsy
                            try_put(draw_queue, ("cursor", dsx, dsy, cur_hex, "draw"))

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
                            try_put(draw_queue, ("cursor", sx, sy, "#777777", "penup"))
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
            cx_buf.clear()
            cy_buf.clear()
            if dragging:
                pyautogui.mouseUp()
                dragging = False
            try_put(draw_queue, ("cursor_hide",))
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
        if DISPLAY_FPS:
            cv2.putText(frame, f"FPS:{fps}", (CAM_W - 70, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

        if result.multi_hand_landmarks:
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

        m = get_mode()
        hints = (
            ["idx=move", "idx+mid=scroll", "mid+th=tap/hold", "tap=click hold=drag", "idx+th=R-click", "spread=zoom", "palm=tab"]
            if m == 1
            else ["relax=pen up", "idx+th=draw", "mid+th=erase", "rng+th=color", "pky+th(hold)=clear", "S=save"]
        )
        for i, h in enumerate(hints):
            cv2.putText(frame, h, (CAM_W - 145, CAM_H - 110 + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

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

    if window_switch_active:
        pyautogui.keyUp("alt")
        window_switch_active = False

    cap.release()
    cv2.destroyAllWindows()

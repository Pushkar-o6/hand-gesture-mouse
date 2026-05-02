import queue
import threading

import pyautogui

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

draw_queue = queue.Queue(maxsize=8)
ctrl_queue = queue.Queue(maxsize=4)
viewer_queue = queue.Queue(maxsize=4)

_mode_lock = threading.Lock()
_current_mode = 1

screen_w, screen_h = pyautogui.size()


def get_mode() -> int:
    # Single int reads are GIL-atomic in CPython; no lock needed
    return _current_mode


def set_mode(mode: int) -> None:
    global _current_mode
    with _mode_lock:
        _current_mode = mode


def toggle_mode() -> int:
    global _current_mode
    with _mode_lock:
        _current_mode = 2 if _current_mode == 1 else 1
        return _current_mode


def try_put(q: queue.Queue, msg) -> None:
    try:
        q.put_nowait(msg)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(msg)
        except queue.Full:
            pass


def try_put_ctrl(msg) -> None:
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

import queue
import threading
from screeninfo import get_monitors

draw_queue = queue.Queue(maxsize=8)
ctrl_queue = queue.Queue(maxsize=4)

_mode_lock = threading.Lock()
_current_mode = 1

def get_virtual_bounds():
    try:
        monitors = get_monitors()
        left = min(m.x for m in monitors)
        top = min(m.y for m in monitors)
        right = max(m.x + m.width for m in monitors)
        bottom = max(m.y + m.height for m in monitors)
        return left, top, right, bottom
    except Exception:
        import ctypes
        user32 = ctypes.windll.user32
        left = user32.GetSystemMetrics(76)
        top = user32.GetSystemMetrics(77)
        w = user32.GetSystemMetrics(78)
        h = user32.GetSystemMetrics(79)
        if w == 0:
            w, h = 1920, 1080
            left, top = 0, 0
        return left, top, left + w, top + h

screen_left, screen_top, screen_right, screen_bottom = get_virtual_bounds()
screen_w = screen_right - screen_left
screen_h = screen_bottom - screen_top


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

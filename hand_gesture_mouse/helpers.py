import math

import numpy as np

from .settings import CAM_MARGIN_X, CAM_MARGIN_Y


class LMPoint:
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x = x
        self.y = y


def lm_norm(lm, idx):
    return lm[idx].x, lm[idx].y


def dist_norm(lm, a, b):
    ax, ay = lm[a].x, lm[a].y
    bx, by = lm[b].x, lm[b].y
    return math.hypot(ax - bx, ay - by)


def hand_scale(lm):
    return dist_norm(lm, 0, 9) + 1e-6


def is_finger_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y


def _angle_deg(a, b, c):
    ab = np.array([a.x - b.x, a.y - b.y], dtype=np.float32)
    cb = np.array([c.x - b.x, c.y - b.y], dtype=np.float32)
    ab_n = np.linalg.norm(ab)
    cb_n = np.linalg.norm(cb)
    if ab_n < 1e-6 or cb_n < 1e-6:
        return 0.0
    cosang = float(np.dot(ab, cb) / (ab_n * cb_n))
    return math.degrees(math.acos(np.clip(cosang, -1.0, 1.0)))


def is_finger_extended(lm, tip, pip, mcp, angle_deg=165.0):
    return _angle_deg(lm[tip], lm[pip], lm[mcp]) >= angle_deg


def count_fingers_up(lm):
    return sum(1 for t, p in [(8, 6), (12, 10), (16, 14), (20, 18)] if is_finger_up(lm, t, p))


def is_fist(lm):
    return count_fingers_up(lm) == 0


def is_open_palm(lm):
    return count_fingers_up(lm) == 4


def is_ily_gesture(lm):
    return (
        is_finger_up(lm, 8, 6)
        and is_finger_up(lm, 20, 18)
        and not is_finger_up(lm, 12, 10)
        and not is_finger_up(lm, 16, 14)
    )


def norm_to_screen(nx, ny, screen_w, screen_h):
    x_lo, x_hi = CAM_MARGIN_X, 1.0 - CAM_MARGIN_X
    y_lo, y_hi = CAM_MARGIN_Y, 1.0 - CAM_MARGIN_Y
    sx = int(np.interp(nx, [x_lo, x_hi], [0, screen_w]))
    sy = int(np.interp(ny, [y_lo, y_hi], [0, screen_h]))
    sx = int(np.clip(sx, 0, screen_w - 1))
    sy = int(np.clip(sy, 0, screen_h - 1))
    return sx, sy

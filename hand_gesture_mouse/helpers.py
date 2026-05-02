import math

import numpy as np

from .settings import CAM_MARGIN_X, CAM_MARGIN_Y


class LMPoint:
    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class OneEuroFilter:
    """Adaptive low-pass filter for pointer smoothing.
    
    Reduces jitter while maintaining responsiveness by adapting the smoothing
    factor based on motion velocity. Standard in interactive systems.
    
    Args:
        freq: Cutoff frequency in Hz (default 120Hz typical for 60fps)
        mincutoff: Minimum cutoff frequency (default 1.0 Hz)
        beta: Velocity coefficient controlling adaptation (default 0.007)
    """
    
    __slots__ = ("freq", "mincutoff", "beta", "last_x", "last_dx", "last_t")
    
    def __init__(self, freq=120.0, mincutoff=1.0, beta=0.007):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.last_x = None
        self.last_dx = 0.0
        self.last_t = None
    
    def _smoothing_factor(self, cutoff, dt):
        """Compute low-pass filtering coefficient."""
        if dt <= 0:
            return 1.0
        r = 2.0 * math.pi * cutoff * dt
        return r / (r + 1.0)
    
    def filter(self, x, t=None):
        """Apply OneEuro filter to value x at time t.
        
        Args:
            x: Current value (float or (x, y) tuple)
            t: Current time (seconds). If None, uses internal timing.
        
        Returns:
            Filtered value (same type as input)
        """
        if t is None:
            if self.last_t is None:
                t = 0.0
            else:
                t = self.last_t + 1.0 / self.freq
        
        if self.last_t is None:
            self.last_t = t
            self.last_x = x
            return x
        
        dt = t - self.last_t
        if dt < 0:
            return x
        
        # Handle both scalar and tuple inputs
        is_tuple = isinstance(x, (tuple, list))
        if is_tuple:
            x = np.array(x, dtype=np.float32)
        
        if self.last_x is None:
            self.last_x = x.copy() if is_tuple else x
            dx = 0.0 if not is_tuple else np.array([0.0, 0.0], dtype=np.float32)
        else:
            if is_tuple:
                dx = (x - self.last_x) / max(dt, 1e-6)
                last_dx = self.last_dx
                edgef = self._smoothing_factor(self.mincutoff, dt)
                dx = last_dx + edgef * (dx - last_dx)
                cutoff = self.mincutoff + self.beta * np.linalg.norm(dx)
                xf = self._smoothing_factor(cutoff, dt)
                x_smooth = self.last_x + xf * (x - self.last_x)
                self.last_x = x_smooth.copy()
            else:
                dx = (x - self.last_x) / max(dt, 1e-6)
                edgef = self._smoothing_factor(self.mincutoff, dt)
                dx = self.last_dx + edgef * (dx - self.last_dx)
                cutoff = self.mincutoff + self.beta * abs(dx)
                xf = self._smoothing_factor(cutoff, dt)
                x_smooth = self.last_x + xf * (x - self.last_x)
                self.last_x = x_smooth
            
            self.last_dx = dx
        
        self.last_t = t
        
        if is_tuple:
            return tuple(self.last_x)
        return self.last_x


def dist_batch(lm, pairs):
    """Compute multiple distances in a single batch operation.
    
    Args:
        lm: Landmark list
        pairs: List of (a, b) index tuples to compute distances for
    
    Returns:
        List of distances in same order as pairs
    """
    coords = np.array([[lm[i].x, lm[i].y] for i in range(len(lm))], dtype=np.float32)
    pair_idx = np.asarray(pairs, dtype=np.intp)
    diffs = coords[pair_idx[:, 0]] - coords[pair_idx[:, 1]]
    return np.linalg.norm(diffs, axis=1).astype(float).tolist()


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


def are_fingers_together(lm, proximity_threshold=0.10):
    """Check if all 4 fingers are joined/close together and extended.
    
    Calculates the centroid of all 4 finger tips and checks if each tip
    is within proximity_threshold distance from the centroid. Also verifies
    that all 4 fingers are extended (tips above PIPs).
    
    Args:
        lm: Landmark list
        proximity_threshold: Maximum distance from centroid for fingers to be "together" (normalized 0-1)
    
    Returns:
        bool: True if all 4 finger tips are extended and close together
    """
    # First check: all 4 fingers must be extended (tips above PIPs)
    finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
    if not all(is_finger_up(lm, tip, pip) for tip, pip in finger_pairs):
        return False
    
    # Get positions of all 4 finger tips: index, middle, ring, pinky
    tip_indices = [8, 12, 16, 20]
    tips = np.array([[lm[i].x, lm[i].y] for i in tip_indices], dtype=np.float32)
    
    # Calculate centroid of all 4 finger tips
    centroid = np.mean(tips, axis=0)
    
    # Check if all tips are within threshold distance from centroid
    distances = np.linalg.norm(tips - centroid, axis=1)
    
    return bool(np.all(distances < proximity_threshold))


def norm_to_screen(nx, ny, screen_w, screen_h):
    x_lo, x_hi = CAM_MARGIN_X, 1.0 - CAM_MARGIN_X
    y_lo, y_hi = CAM_MARGIN_Y, 1.0 - CAM_MARGIN_Y
    sx = int(np.interp(nx, [x_lo, x_hi], [0, screen_w]))
    sy = int(np.interp(ny, [y_lo, y_hi], [0, screen_h]))
    sx = int(np.clip(sx, 0, screen_w - 1))
    sy = int(np.clip(sy, 0, screen_h - 1))
    return sx, sy

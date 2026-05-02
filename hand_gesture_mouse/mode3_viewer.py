import queue
import time

import cv2
import numpy as np

from .helpers import OneEuroFilter
from .state import viewer_queue

VIEW_W, VIEW_H = 760, 540


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def _project(pts3d, f=480, cx=VIEW_W // 2, cy=VIEW_H // 2):
    z = np.maximum(pts3d[:, 2] + 4.2, 0.4)
    x = (pts3d[:, 0] * f / z) + cx
    y = (pts3d[:, 1] * f / z) + cy
    return np.stack([x, y], axis=1).astype(np.int32)


CUBE_VERTS = np.array(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    dtype=np.float32,
)

CUBE_FACES = [
    (0, 1, 2, 3),
    (4, 7, 6, 5),
    (0, 4, 5, 1),
    (3, 2, 6, 7),
    (1, 5, 6, 2),
    (0, 3, 7, 4),
]

CUBE_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]

FACE_BASE_COLORS = np.array(
    [
        [60, 120, 255],
        [70, 210, 255],
        [90, 230, 160],
        [255, 190, 90],
        [210, 110, 255],
        [255, 110, 140],
    ],
    dtype=np.float32,
)


def _drain_messages():
    messages = []
    while True:
        try:
            messages.append(viewer_queue.get_nowait())
        except queue.Empty:
            return messages


def _background():
    y = np.linspace(0.0, 1.0, VIEW_H, dtype=np.float32)[:, None]
    top = np.array([18, 22, 30], dtype=np.float32)
    bottom = np.array([4, 7, 12], dtype=np.float32)
    row = top * (1.0 - y) + bottom * y
    return np.repeat(row[:, None, :], VIEW_W, axis=1).astype(np.uint8)


def _draw_shadow(canvas, pts2d, verts):
    center = pts2d.mean(axis=0).astype(int)
    width = int(np.clip(np.ptp(pts2d[:, 0]) * 0.44, 45, 180))
    height = int(np.clip(width * 0.22, 12, 45))
    y = int(np.clip(center[1] + np.ptp(pts2d[:, 1]) * 0.48 + 32 + verts[:, 2].mean() * 3, 80, VIEW_H - 45))
    overlay = canvas.copy()
    cv2.ellipse(overlay, (int(center[0]), y), (width, height), 0, 0, 360, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.32, canvas, 0.68, 0, dst=canvas)


def _face_normal(face_verts):
    a, b, c = face_verts[:3]
    n = np.cross(b - a, c - a)
    norm = float(np.linalg.norm(n))
    if norm < 1e-6:
        return np.array([0, 0, 1], dtype=np.float32)
    return (n / norm).astype(np.float32)


def _draw_cube(canvas, yaw, pitch, roll, scale, tx, ty, started):
    idle = time.perf_counter() - started
    hover = np.sin(idle * 2.3) * 0.08
    idle_yaw = np.sin(idle * 0.55) * 0.035
    idle_pitch = np.cos(idle * 0.43) * 0.025

    r_mat = _rot_y(yaw + idle_yaw) @ _rot_x(pitch + idle_pitch) @ _rot_z(roll)
    verts = (CUBE_VERTS * scale) @ r_mat.T
    verts[:, 0] += tx * 1.45
    verts[:, 1] += ty * 1.25 + hover
    pts2d = _project(verts)

    _draw_shadow(canvas, pts2d, verts)

    light_dir = np.array([-0.35, -0.65, 0.68], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)

    order = sorted(range(len(CUBE_FACES)), key=lambda i: float(verts[list(CUBE_FACES[i]), 2].mean()))
    for face_idx in order:
        face = CUBE_FACES[face_idx]
        face_verts = verts[list(face)]
        normal = _face_normal(face_verts)
        shade = float(np.clip(0.42 + 0.58 * np.dot(normal, light_dir), 0.22, 1.0))
        color = np.clip(FACE_BASE_COLORS[face_idx] * shade, 0, 255).astype(np.uint8)
        pts = pts2d[list(face)]
        cv2.fillConvexPoly(canvas, pts, tuple(int(v) for v in color), cv2.LINE_AA)
        cv2.polylines(canvas, [pts], True, (230, 240, 255), 1, cv2.LINE_AA)

    for a, b in CUBE_EDGES:
        cv2.line(canvas, tuple(pts2d[a]), tuple(pts2d[b]), (245, 248, 255), 1, cv2.LINE_AA)

    for pt in pts2d:
        cv2.circle(canvas, tuple(pt), 3, (255, 255, 255), -1, cv2.LINE_AA)


def _draw_hud(canvas, yaw, pitch, roll, scale, tx, ty):
    cv2.putText(
        canvas,
        "MODE 3  FLOATING CUBE",
        (18, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (220, 235, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"yaw {np.degrees(yaw):5.1f}   pitch {np.degrees(pitch):5.1f}   roll {np.degrees(roll):5.1f}",
        (18, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (145, 165, 190),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"scale {scale:.2f}   x {tx:+.2f}   y {ty:+.2f}",
        (18, 73),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (145, 165, 190),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "spread wrists: zoom   move midpoint: float   tilt hands: rotate   fists: reset   ILY+palm: exit",
        (18, VIEW_H - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (100, 118, 138),
        1,
        cv2.LINE_AA,
    )


def viewer_thread():
    yaw = pitch = roll = 0.0
    scale = 1.0
    tx = ty = 0.0
    active = False
    window_visible = False
    started = time.perf_counter()

    oef_yaw = OneEuroFilter(freq=60, mincutoff=1.35, beta=0.018)
    oef_pitch = OneEuroFilter(freq=60, mincutoff=1.35, beta=0.018)
    oef_roll = OneEuroFilter(freq=60, mincutoff=1.25, beta=0.015)
    oef_scale = OneEuroFilter(freq=60, mincutoff=1.8, beta=0.01)
    oef_tx = OneEuroFilter(freq=60, mincutoff=1.6, beta=0.018)
    oef_ty = OneEuroFilter(freq=60, mincutoff=1.6, beta=0.018)

    while True:
        messages = _drain_messages()
        latest_transform = None
        for msg in messages:
            kind = msg[0]
            if kind == "active":
                active = bool(msg[1])
                if active:
                    started = time.perf_counter()
                elif window_visible:
                    cv2.destroyWindow("3D Viewer")
                    window_visible = False
            elif kind == "transform":
                latest_transform = msg
            elif kind == "reset":
                yaw = pitch = roll = 0.0
                scale = 1.0
                tx = ty = 0.0
        if latest_transform is not None:
            _, raw_yaw, raw_pitch, raw_roll, raw_scale, raw_tx, raw_ty = latest_transform
            t = time.perf_counter()
            yaw = oef_yaw.filter(raw_yaw, t)
            pitch = oef_pitch.filter(raw_pitch, t)
            roll = oef_roll.filter(raw_roll, t)
            scale = oef_scale.filter(raw_scale, t)
            tx = oef_tx.filter(raw_tx, t)
            ty = oef_ty.filter(raw_ty, t)

        if not active:
            time.sleep(0.025)
            continue

        if not window_visible:
            cv2.namedWindow("3D Viewer", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("3D Viewer", VIEW_W, VIEW_H)
            window_visible = True

        canvas = _background()
        _draw_cube(canvas, yaw, pitch, roll, scale, tx, ty, started)
        _draw_hud(canvas, yaw, pitch, roll, scale, tx, ty)

        cv2.imshow("3D Viewer", canvas)
        if cv2.waitKey(1) & 0xFF == 27:
            active = False
            cv2.destroyWindow("3D Viewer")
            window_visible = False

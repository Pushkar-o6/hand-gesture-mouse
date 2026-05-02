import math
import queue
import time
import tkinter as tk

import cv2
import numpy as np

from .settings import (
    EDITOR_COLORS_HEX,
    ERASER_SIZE,
    TRANSPARENT_KEY,
)
from .state import ctrl_queue, draw_queue, screen_h


class ScreenOverlay:
    def __init__(self):
        self.root = tk.Tk()
        self._running = True
        self._poll_after_id = None
        self.root.title("Screen Editor Overlay")
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-transparentcolor", TRANSPARENT_KEY)
        self.root.configure(bg=TRANSPARENT_KEY)
        self.root.overrideredirect(True)

        self.canvas = tk.Canvas(
            self.root,
            bg=TRANSPARENT_KEY,
            highlightthickness=0,
            cursor="none",
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<KeyPress-s>", self.save_drawing)
        self.canvas.bind("<KeyPress-c>", lambda e: self.clear_canvas())
        self.canvas.bind("<Escape>", self.quit)
        self.canvas.focus_set()

        self.hud_id = None
        self.cursor_ids = []
        self.palette_ids = []
        self.strokes = []
        self.color_idx = 0
        self.live_stroke_ids = {}

        self._draw_palette()
        self._draw_hud("Mode 1: Mouse Control", "#FF8C00")
        self._poll_after_id = self.root.after(8, self._poll)

    def _poll(self):
        if not self._running:
            return

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

        if self._running:
            self._poll_after_id = self.root.after(8, self._poll)

    def _handle(self, msg):
        kind = msg[0]

        if kind == "draw":
            if len(msg) == 8:
                _, x1, y1, x2, y2, color, size, stroke_id = msg
            else:
                _, x1, y1, x2, y2, color, size = msg
                stroke_id = None

            iid = self.canvas.create_line(
                x1,
                y1,
                x2,
                y2,
                fill=color,
                width=size,
                capstyle=tk.ROUND,
                joinstyle=tk.ROUND,
                smooth=True,
            )
            self.strokes.append(iid)
            if stroke_id is not None:
                self.live_stroke_ids.setdefault(stroke_id, []).append(iid)

        elif kind == "draw_batch":
            _, stroke_id, points, color, size = msg
            if len(points) >= 2:
                flat = [v for p in points for v in p]
                iid = self.canvas.create_line(
                    *flat,
                    fill=color,
                    width=size,
                    capstyle=tk.ROUND,
                    joinstyle=tk.ROUND,
                    smooth=True,
                    splinesteps=12,
                )
                self.strokes.append(iid)
                if stroke_id is not None:
                    self.live_stroke_ids.setdefault(stroke_id, []).append(iid)

        elif kind == "stroke_begin":
            _, stroke_id = msg
            self.live_stroke_ids[stroke_id] = []

        elif kind == "stroke_finalize":
            _, stroke_id, points, color, size = msg
            self._finalize_stroke(stroke_id, points, color, size)

        elif kind == "erase":
            _, ex, ey, sz = msg
            r = sz // 2
            iid = self.canvas.create_oval(
                ex - r,
                ey - r,
                ex + r,
                ey + r,
                fill=TRANSPARENT_KEY,
                outline=TRANSPARENT_KEY,
            )
            self.strokes.append(iid)

        elif kind == "clear":
            self.clear_canvas()

        elif kind == "mode":
            _, m = msg
            if m == 1:
                for iid in self.strokes:
                    self.canvas.itemconfigure(iid, state="hidden")
                self._draw_hud("Mode 1: Mouse Control", "#FF8C00")
            elif m == 2:
                for iid in self.strokes:
                    self.canvas.itemconfigure(iid, state="normal")
                self._draw_hud("Mode 2: Screen Editor  [pinch=draw  2-pinch=erase]", "#00D264")
            else:
                for iid in self.strokes:
                    self.canvas.itemconfigure(iid, state="hidden")
                self._clear_cursor()
                self._draw_hud("Mode 3: 3D Object Viewer", "#64C8FF")

        elif kind == "cursor":
            _, cx, cy, color, style = msg
            self._draw_cursor(cx, cy, color, style)

        elif kind == "cursor_hide":
            self._clear_cursor()

        elif kind == "color":
            _, idx = msg
            self.color_idx = idx
            self._draw_palette()

        elif kind == "save":
            self.save_drawing()

        elif kind == "quit":
            self._running = False
            if self._poll_after_id is not None:
                try:
                    self.root.after_cancel(self._poll_after_id)
                except tk.TclError:
                    pass
                self._poll_after_id = None
            self.root.quit()

    def _clear_cursor(self):
        for iid in self.cursor_ids:
            self.canvas.delete(iid)
        self.cursor_ids = []

    def _draw_cursor(self, cx, cy, color, style):
        self._clear_cursor()
        ids = []

        if style == "draw":
            r = 5
            ids.append(self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=color, outline="black", width=1))
            arm = 14
            for x1, y1, x2, y2 in [
                (cx - arm, cy, cx - r - 2, cy),
                (cx + r + 2, cy, cx + arm, cy),
                (cx, cy - arm, cx, cy - r - 2),
                (cx, cy + r + 2, cx, cy + arm),
            ]:
                ids.append(self.canvas.create_line(x1, y1, x2, y2, fill=color, width=2))

        elif style == "penup":
            r = 8
            ids.append(self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#777777", width=1, dash=(3, 3)))
            arm = 18
            for x1, y1, x2, y2 in [
                (cx - arm, cy, cx - r - 2, cy),
                (cx + r + 2, cy, cx + arm, cy),
                (cx, cy - arm, cx, cy - r - 2),
                (cx, cy + r + 2, cx, cy + arm),
            ]:
                ids.append(self.canvas.create_line(x1, y1, x2, y2, fill="#777777", width=1, dash=(3, 3)))

        elif style == "erase":
            r = ERASER_SIZE
            ids.append(self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#aaaaaa", width=2, dash=(4, 4)))
            ids.append(self.canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, outline="#aaaaaa"))

        self.cursor_ids = ids

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
        if shape_kind == "line":
            (x1, y1), (x2, y2) = geom
            iid = self.canvas.create_line(
                x1,
                y1,
                x2,
                y2,
                fill=color,
                width=size,
                capstyle=tk.ROUND,
                joinstyle=tk.ROUND,
                smooth=False,
            )
            self.strokes.append(iid)

        elif shape_kind == "rect":
            rect_pts = geom
            flat = [v for p in rect_pts for v in p]
            iid = self.canvas.create_polygon(
                *flat,
                outline=color,
                fill="",
                width=size,
                smooth=False,
            )
            self.strokes.append(iid)

        elif shape_kind == "circle":
            cx, cy, r = geom
            iid = self.canvas.create_oval(
                cx - r,
                cy - r,
                cx + r,
                cy + r,
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
            return "raw", None

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
                return "line", ((float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1])))

        contour = pts.astype(np.int32).reshape(-1, 1, 2)
        area = abs(cv2.contourArea(contour))
        if area < 80:
            return "raw", None

        peri = cv2.arcLength(contour, True)
        circularity = float((4.0 * math.pi * area) / (peri * peri + 1e-6))
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        radius = max(1e-6, float(radius))
        dists = np.linalg.norm(pts - np.array([cx, cy], dtype=np.float32), axis=1)
        radial_std_ratio = float(np.std(dists) / radius)
        aspect_ratio = float(min(w, h) / max(w, h, 1e-6))

        if circularity > 0.74 and radial_std_ratio < 0.22 and aspect_ratio > 0.72:
            return "circle", (float(cx), float(cy), float(radius))

        approx = cv2.approxPolyDP(contour, 0.03 * peri, True)

        if len(approx) == 4:
            quad = [(float(p[0][0]), float(p[0][1])) for p in approx]
            if self._quad_is_rect(quad):
                ordered = self._order_quad(quad)
                rect_area = max(1.0, cv2.contourArea(np.array(ordered, dtype=np.float32)))
                fill_ratio = area / rect_area
                edge_support = self._rectangle_edge_support(pts, ordered)
                if fill_ratio > 0.68 and edge_support > 0.74:
                    return "rect", ordered

        if circularity > 0.70 and radial_std_ratio < 0.20 and aspect_ratio > 0.80:
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            return "circle", (float(cx), float(cy), float(radius))

        return "raw", None

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
        tol = max(
            4.0,
            0.025
            * float(
                np.hypot(
                    pts[:, 0].max() - pts[:, 0].min(),
                    pts[:, 1].max() - pts[:, 1].min(),
                )
            ),
        )
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
            14,
            14,
            text=text,
            anchor="nw",
            fill=color,
            font=("Helvetica", 13, "bold"),
        )

    def _draw_palette(self):
        for iid in self.palette_ids:
            self.canvas.delete(iid)
        self.palette_ids.clear()
        y = screen_h - 22
        for i, hex_c in enumerate(EDITOR_COLORS_HEX):
            x = 22 + i * 30
            r = 12 if i == self.color_idx else 9
            out = "white" if i == self.color_idx else hex_c
            self.palette_ids.append(
                self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=hex_c, outline=out, width=2)
            )

    def save_drawing(self, event=None):
        try:
            from PIL import ImageGrab

            fname = f"drawing_{int(time.time())}.png"
            ImageGrab.grab().save(fname)
            print(f"[Saved] {fname}")
            self._draw_hud(f"Saved: {fname}", "#00FF88")
            self.root.after(2000, lambda: self._draw_hud("Mode 2: Screen Editor", "#00D264"))
        except ImportError:
            print("[Save] pip install Pillow")

    def quit(self, event=None):
        self._running = False
        if self._poll_after_id is not None:
            try:
                self.root.after_cancel(self._poll_after_id)
            except tk.TclError:
                pass
            self._poll_after_id = None
        self.root.quit()

    def run(self):
        self.root.mainloop()

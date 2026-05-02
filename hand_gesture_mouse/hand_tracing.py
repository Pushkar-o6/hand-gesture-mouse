import pathlib
import queue
import sys
import threading


# Support both:
# 1) python -m hand_gesture_mouse.hand_tracing
# 2) python hand_tracing.py  (from inside hand_gesture_mouse folder)
if __package__ in (None, ""):
    sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
    from hand_gesture_mouse.controller import webcam_thread
    from hand_gesture_mouse.settings import CAM_H, CAM_W
    from hand_gesture_mouse.overlay import ScreenOverlay
    from hand_gesture_mouse.mode3_viewer import viewer_thread
else:
    from .controller import webcam_thread
    from .settings import CAM_H, CAM_W
    from .overlay import ScreenOverlay
    from .mode3_viewer import viewer_thread


def _capture_thread(frame_q: queue.Queue):
    import cv2

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        try:
            frame_q.get_nowait()
        except queue.Empty:
            pass
        try:
            frame_q.put_nowait(frame)
        except queue.Full:
            pass


def main():
    frame_q = queue.Queue(maxsize=2)
    cam_thread = threading.Thread(target=_capture_thread, args=(frame_q,), daemon=True)
    cam_thread.start()

    cam_thread = threading.Thread(target=webcam_thread, args=(frame_q,), daemon=True)
    cam_thread.start()

    viewer = threading.Thread(target=viewer_thread, daemon=True)
    viewer.start()

    overlay = ScreenOverlay()
    overlay.run()


if __name__ == "__main__":
    main()

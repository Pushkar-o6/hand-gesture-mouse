import pathlib
import sys
import threading


# Support both:
# 1) python -m hand_gesture_mouse.hand_tracing
# 2) python hand_tracing.py  (from inside hand_gesture_mouse folder)
if __package__ in (None, ""):
    sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
    from hand_gesture_mouse.controller import webcam_thread
    from hand_gesture_mouse.overlay import ScreenOverlay
else:
    from .controller import webcam_thread
    from .overlay import ScreenOverlay


def main():
    cam_thread = threading.Thread(target=webcam_thread, daemon=True)
    cam_thread.start()

    overlay = ScreenOverlay()
    overlay.run()


if __name__ == "__main__":
    main()
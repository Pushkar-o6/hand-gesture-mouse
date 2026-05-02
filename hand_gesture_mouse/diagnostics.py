import importlib
import sys


REQUIRED_MODULES = ("cv2", "mediapipe", "numpy", "pyautogui")
OPTIONAL_MODULES = ("PIL", "pycaw", "comtypes", "pygetwindow")


def _check_import(name):
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "installed")
        return True, str(version)
    except Exception as exc:
        return False, str(exc)


def run() -> int:
    ok = True
    print("Hand Gesture Mouse diagnostics")
    print(f"Python: {sys.version.split()[0]}")

    for name in REQUIRED_MODULES:
        present, info = _check_import(name)
        ok = ok and present
        status = "OK" if present else "MISSING"
        print(f"{status:8s} {name:12s} {info}")

    for name in OPTIONAL_MODULES:
        present, info = _check_import(name)
        status = "OK" if present else "optional"
        print(f"{status:8s} {name:12s} {info}")

    if ok:
        import cv2
        import pyautogui

        print(f"Screen: {pyautogui.size()}")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        cam_ok = cap.isOpened()
        print(f"Camera: {'OK' if cam_ok else 'not available'}")
        if cam_ok:
            cap.release()
        ok = ok and cam_ok

    return 0 if ok else 1


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()

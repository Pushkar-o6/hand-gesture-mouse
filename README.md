# hand-gesture-mouse

This project now uses an OpenCV-native hand tracker (no `mediapipe` runtime dependency)
for lower startup overhead and lightweight CPU inference at 360p.

Install dependencies:

```bash
pip install opencv-python pyautogui numpy pillow
```

Run:

```bash
python -m hand_gesture_mouse.hand_tracing
```
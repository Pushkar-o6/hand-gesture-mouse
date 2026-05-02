import argparse
import time

import cv2
import mediapipe as mp
import numpy as np

from .gesture_ml import extract_features
from .helpers import LMPoint
from .settings import CAM_H, CAM_W, MAX_NUM_HANDS, PROC_H, PROC_W

LABEL_KEYS = {
    ord("1"): "MOVE",
    ord("2"): "SCROLL",
    ord("3"): "CLICK",
    ord("4"): "RIGHT_CLICK",
    ord("5"): "ZOOM",
    ord("6"): "DRAW",
    ord("7"): "ERASE",
    ord("8"): "COLOR",
    ord("9"): "CLEAR",
    ord("0"): "MODE_SWITCH",
}


def record_samples(out_path: str, max_per_label: int) -> None:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        model_complexity=0,
        min_detection_confidence=0.60,
        min_tracking_confidence=0.60,
    )

    features = []
    labels = []
    counts = {label: 0 for label in LABEL_KEYS.values()}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        small = cv2.resize(frame, (PROC_W, PROC_H))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            lm = [LMPoint(pt.x, pt.y) for pt in hand.landmark]
            for conn in mp_hands.HAND_CONNECTIONS:
                a, b = conn
                ax = int(lm[a].x * CAM_W)
                ay = int(lm[a].y * CAM_H)
                bx = int(lm[b].x * CAM_W)
                by = int(lm[b].y * CAM_H)
                cv2.line(frame, (ax, ay), (bx, by), (80, 80, 220), 1)
            for pt in lm:
                cv2.circle(frame, (int(pt.x * CAM_W), int(pt.y * CAM_H)), 2, (0, 255, 180), -1)

        cv2.rectangle(frame, (0, 0), (CAM_W, 44), (25, 25, 25), -1)
        cv2.putText(
            frame,
            "Record: 1-9 labels | 0=MODE | ESC=quit",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
        )

        y = 64
        for label, count in counts.items():
            cv2.putText(
                frame,
                f"{label}: {count}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (160, 160, 160),
                1,
            )
            y += 18

        cv2.imshow("Gesture Recorder", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        if key in LABEL_KEYS and res.multi_hand_landmarks:
            label = LABEL_KEYS[key]
            if counts[label] < max_per_label:
                hand = res.multi_hand_landmarks[0]
                lm = [LMPoint(pt.x, pt.y) for pt in hand.landmark]
                feat = extract_features(lm)
                features.append(feat)
                labels.append(label)
                counts[label] += 1
                time.sleep(0.05)

    cap.release()
    cv2.destroyAllWindows()

    if features:
        arr_x = np.stack(features, axis=0)
        arr_y = np.asarray(labels)
        np.savez_compressed(out_path, features=arr_x, labels=arr_y)
        print(f"[Saved] {out_path} ({len(arr_y)} samples)")
    else:
        print("[Info] No samples recorded.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record gesture samples for KNN model.")
    parser.add_argument("--out", default="gesture_samples.npz", help="Output dataset path")
    parser.add_argument("--max-per-label", type=int, default=120, help="Max samples per label")
    args = parser.parse_args()
    record_samples(args.out, args.max_per_label)


if __name__ == "__main__":
    main()

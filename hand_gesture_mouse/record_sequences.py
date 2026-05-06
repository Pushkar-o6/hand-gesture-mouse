import argparse
import time
import os
import cv2
import mediapipe as mp
import numpy as np
from collections import deque

from .gesture_ml import extract_features
from .settings import CAM_H, CAM_W
from .helpers import LMPoint

LABELS = ["idle", "swipe_left", "swipe_right", "scroll_up", "scroll_down"]
SEQ_LEN = 15

def record_data():
    print("========================================")
    print(" TEMPORAL GESTURE DATA RECORDER")
    print(f" Sequence Length: {SEQ_LEN} frames")
    print("========================================")
    print("Press '0' -> idle")
    print("Press '1' -> swipe_left")
    print("Press '2' -> swipe_right")
    print("Press '3' -> scroll_up")
    print("Press '4' -> scroll_down")
    print("Press 's' -> Save and Quit")
    print("Press 'q' -> Discard and Quit")
    print("========================================\n")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    dataset_X = []
    dataset_y = []
    
    current_label = None
    recording_buffer = []

    print("[Ready] Press 0-4 to trigger a recording snapshot for that class.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        display_frame = frame.copy()
        
        # If actively recording
        if current_label is not None:
            cv2.putText(display_frame, f"RECORDING '{LABELS[current_label]}' ({len(recording_buffer)}/{SEQ_LEN})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if res.multi_hand_landmarks:
                raw_lm = res.multi_hand_landmarks[0]
                # Convert to our LMPoint structure
                lm_points = [LMPoint(pt.x * CAM_W, pt.y * CAM_H, pt.z) for pt in raw_lm.landmark]
                
                # Extract ML-friendly features (normalized against hand size)
                feats = extract_features(lm_points)
                recording_buffer.append(feats)
                
            # If we reached our sequence length
            if len(recording_buffer) == SEQ_LEN:
                dataset_X.append(recording_buffer.copy())
                dataset_y.append(current_label)
                print(f"[Captured] '{LABELS[current_label]}' | Total Samples: {len(dataset_X)}")
                recording_buffer.clear()
                current_label = None
        else:
            cv2.putText(display_frame, f"Samples stored: {len(dataset_X)}. Press 0-4 to record.", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            if res.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(display_frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Gesture Recorder", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if current_label is None:
            if key == ord('0'): current_label = 0
            if key == ord('1'): current_label = 1
            if key == ord('2'): current_label = 2
            if key == ord('3'): current_label = 3
            if key == ord('4'): current_label = 4
            
        if key == ord('s'):
            if len(dataset_X) > 0:
                out_path = "sequence_samples.npz"
                np.savez_compressed(out_path, X=np.array(dataset_X), y=np.array(dataset_y))
                print(f"[Saved] Wrote {len(dataset_X)} samples to {out_path}.")
            else:
                print("No samples to save.")
            break
        elif key == ord('q'):
            print("Quit without saving.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    record_data()
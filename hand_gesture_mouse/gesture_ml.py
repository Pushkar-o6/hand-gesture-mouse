import os
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

# Try to import onnxruntime for the temporal model
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

class TemporalGestureEngine:
    def __init__(self, model_path="temporal_gesture_model.onnx", seq_len=15):
        self.seq_len = seq_len
        self.model_path = model_path
        self.buffer = deque(maxlen=seq_len)
        self.session = None
        self.labels = ["idle", "swipe_left", "swipe_right", "scroll_up", "scroll_down"]
        
        if HAS_ONNX and os.path.exists(self.model_path):
            try:
                self.session = ort.InferenceSession(self.model_path)
            except Exception as e:
                print(f"Failed to load ONNX model: {e}")

    def add_frame(self, lm):
        """Extract and normalize features for the current frame, add to buffer."""
        feat = extract_features(lm)
        self.buffer.append(feat)

    def predict(self) -> str:
        """Run inference on the buffered sequence."""
        if len(self.buffer) < self.seq_len:
            return "idle"

        # If no model is trained/loaded yet, return idle to fallback to heuristics
        if not self.session:
            return "idle"

        # Shape: (1, seq_len, num_features)
        seq_arr = np.array(self.buffer, dtype=np.float32)[np.newaxis, ...]
        
        input_name = self.session.get_inputs()[0].name
        outs = self.session.run(None, {input_name: seq_arr})
        
        # Assume outs[0] is logits/probs of shape (1, num_classes)
        probs = outs[0][0]
        pred_idx = int(np.argmax(probs))
        
        if probs[pred_idx] > 0.75: # Confidence threshold
            return self.labels[pred_idx]
        return "idle"

    def clear(self):
        self.buffer.clear()

@dataclass
class GestureKNN:
    features: np.ndarray
    labels: np.ndarray
    k: int = 5

    def predict(self, feat: np.ndarray) -> str:
        if self.features.size == 0:
            return ""
        feat = np.asarray(feat, dtype=np.float32).reshape(1, -1)
        dists = np.linalg.norm(self.features - feat, axis=1)
        k = min(self.k, len(dists))
        idx = np.argpartition(dists, k - 1)[:k]
        votes = {}
        for i in idx:
            lab = str(self.labels[i])
            votes[lab] = votes.get(lab, 0) + 1
        return max(votes.items(), key=lambda x: x[1])[0]

    def save(self, path: str) -> None:
        np.savez_compressed(path, features=self.features, labels=self.labels, k=self.k)

    @staticmethod
    def load(path: str):
        if not os.path.exists(path):
            return None
        data = np.load(path, allow_pickle=True)
        features = np.asarray(data["features"], dtype=np.float32)
        labels = np.asarray(data["labels"])
        k = int(data["k"]) if "k" in data else 5
        return GestureKNN(features=features, labels=labels, k=k)


def extract_features(lm, scale=None):
    base = lm[0]
    if scale is None:
        dx = lm[9].x - base.x
        dy = lm[9].y - base.y
        scale = float(np.hypot(dx, dy)) + 1e-6
    feats = []
    for pt in lm:
        feats.append((pt.x - base.x) / scale)
        feats.append((pt.y - base.y) / scale)
    return np.asarray(feats, dtype=np.float32)

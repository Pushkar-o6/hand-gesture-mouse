import os
from dataclasses import dataclass

import numpy as np


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

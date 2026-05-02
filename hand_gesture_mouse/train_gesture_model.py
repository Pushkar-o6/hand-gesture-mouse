import argparse

import numpy as np

from .gesture_ml import GestureKNN


def train_model(data_path: str, out_path: str, k: int) -> None:
    data = np.load(data_path, allow_pickle=True)
    features = np.asarray(data["features"], dtype=np.float32)
    labels = np.asarray(data["labels"])

    model = GestureKNN(features=features, labels=labels, k=k)
    model.save(out_path)
    print(f"[Saved] {out_path} (k={k}, samples={len(labels)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple KNN gesture model.")
    parser.add_argument("--data", default="gesture_samples.npz", help="Dataset .npz path")
    parser.add_argument("--out", default="gesture_model.npz", help="Output model path")
    parser.add_argument("--k", type=int, default=5, help="K neighbors")
    args = parser.parse_args()
    train_model(args.data, args.out, args.k)


if __name__ == "__main__":
    main()

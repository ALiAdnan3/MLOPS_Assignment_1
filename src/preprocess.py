"""Stage 2 - preprocess: normalize pixels to [0, 1] and split train/val."""
import os

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"


def normalize(x):
    # Per-image min-max scaling: stretch every image to the full [0, 1] range
    x = x.astype("float32")
    x_min = x.min(axis=(1, 2), keepdims=True)
    x_max = x.max(axis=(1, 2), keepdims=True)
    return (x - x_min) / np.maximum(x_max - x_min, 1e-7)


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["preprocess"]

    train = np.load(os.path.join(RAW_DIR, "train.npz"))
    test = np.load(os.path.join(RAW_DIR, "test.npz"))

    x_train, x_val, y_train, y_val = train_test_split(
        normalize(train["x"]), train["y"],
        test_size=params["val_size"], random_state=params["seed"], stratify=train["y"],
    )
    x_test, y_test = normalize(test["x"]), test["y"]

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(PROCESSED_DIR, "train.npz"), x=x_train, y=y_train)
    np.savez_compressed(os.path.join(PROCESSED_DIR, "val.npz"), x=x_val, y=y_val)
    np.savez_compressed(os.path.join(PROCESSED_DIR, "test.npz"), x=x_test, y=y_test)
    print(f"Saved processed data -> {PROCESSED_DIR}/")
    print(f"Shapes: train={x_train.shape} val={x_val.shape} test={x_test.shape}")


if __name__ == "__main__":
    main()

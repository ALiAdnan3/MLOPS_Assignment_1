"""Stage 1 - prepare: download Fashion-MNIST and save raw arrays to data/raw/."""
import os

import numpy as np
from tensorflow import keras

RAW_DIR = "data/raw"


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    np.savez_compressed(os.path.join(RAW_DIR, "train.npz"), x=x_train, y=y_train)
    np.savez_compressed(os.path.join(RAW_DIR, "test.npz"), x=x_test, y=y_test)
    print(f"Saved raw data -> {RAW_DIR}/  train={x_train.shape}  test={x_test.shape}")


if __name__ == "__main__":
    main()

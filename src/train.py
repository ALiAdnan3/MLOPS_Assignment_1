"""Stage 3 - train: build and train the ANN, save models/model.h5 + models/history.csv."""
import os

import numpy as np
import yaml
from tensorflow import keras

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"


def build_model(p):
    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(p["dense_units"], activation="relu"),
        keras.layers.Dropout(p["dropout_rate"]),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=p["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    with open("params.yaml") as f:
        p = yaml.safe_load(f)["train"]
    keras.utils.set_random_seed(p["seed"])

    train = np.load(os.path.join(PROCESSED_DIR, "train.npz"))
    val = np.load(os.path.join(PROCESSED_DIR, "val.npz"))

    os.makedirs(MODEL_DIR, exist_ok=True)
    model = build_model(p)
    model.fit(
        train["x"], train["y"],
        validation_data=(val["x"], val["y"]),
        epochs=p["epochs"],
        batch_size=p["batch_size"],
        callbacks=[keras.callbacks.CSVLogger(os.path.join(MODEL_DIR, "history.csv"))],
        verbose=2,
    )
    model.save(os.path.join(MODEL_DIR, "model.h5"))
    print(f"Saved model -> {MODEL_DIR}/model.h5")


if __name__ == "__main__":
    main()

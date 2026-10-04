"""Stage 4 - evaluate: test loss/accuracy -> metrics.json, confusion matrix -> confusion_matrix.png."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow import keras

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def main():
    model = keras.models.load_model(os.path.join("models", "model.h5"))
    test = np.load(os.path.join("data", "processed", "test.npz"))

    loss, acc = model.evaluate(test["x"], test["y"], verbose=0)
    y_pred = model.predict(test["x"], verbose=0).argmax(axis=1)

    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(confusion_matrix(test["y"], y_pred), display_labels=CLASS_NAMES).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title(f"Fashion-MNIST ANN - test accuracy {acc:.4f}")
    fig.tight_layout()
    fig.savefig("confusion_matrix.png")

    metrics = {"test_loss": round(float(loss), 4), "test_accuracy": round(float(acc), 4)}
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(metrics)


if __name__ == "__main__":
    main()

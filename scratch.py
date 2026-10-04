# Scratch file: quick experiments while exploring the dataset.
# Obsolete - will be removed with `git rm` (Part A8).
from tensorflow import keras

(x_train, y_train), _ = keras.datasets.fashion_mnist.load_data()
print(x_train.shape, y_train[:10])

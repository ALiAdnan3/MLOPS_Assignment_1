# fashion-ann-pipeline

End-to-end Fashion-MNIST classificaton pipeline using a TensorFlow ANN,
versioned with Git and DVC (Google Drive remote).

## Stack

- TensorFlow / Keras (fully-connected ANN, not a CNN)
- DVC for data and model versioning
- Google Drive as the DVC remote

## Goal

Reach at least 85% test accuracy, reproducible with a single `dvc repro`.

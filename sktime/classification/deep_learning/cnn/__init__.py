"""CNN deep learning time series classifier models.

This subpackage provides Convolutional Neural Network (CNN) based time series
classifier in TensorFlow and PyTorch backends.
"""

__all__ = [
    "CNNClassifier",
    "CNNClassifierTorch",
]

from sktime.classification.deep_learning.cnn._cnn_tf import CNNClassifier
from sktime.classification.deep_learning.cnn._cnn_torch import CNNClassifierTorch

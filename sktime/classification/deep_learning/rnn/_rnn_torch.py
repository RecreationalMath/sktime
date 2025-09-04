"""PyTorch implementation of Time Recurrent Neural Network (RNN) for classification."""

__author__ = ["RecreationMath"]
__all__ = ["SimpleRNNClassifierTorch"]

from sklearn.utils import check_random_state

from sktime.classification.deep_learning.base import BaseDeepClassifier
from sktime.networks.rnn._rnn_torch import RNNNetwork
from sktime.utils.dependencies import _check_dl_dependencies



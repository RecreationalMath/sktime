"""Recurrent Neural Network for Classification and Regression."""

__author__ = ["RecreationalMath"]

from sktime.utils.dependencies import _check_dl_dependencies

if _check_dl_dependencies("torch", severity="none"):
    import torch.nn as nn

class PyTorchRNNNetwork(nn.Module):
    """PyTorch implementation of a Recurrent Neural Network (RNN) for classification and regression."""

    def __init__(
            self, 
            input_shape, 
            num_classes=2, 
            hidden_dims=None, 
            activation=None,    
            # nonlinearity – The non-linearity to use. Can be either 'tanh' or 'relu'. Default: 'tanh'
            dropout=0.0,    # done
            use_bias=False, # done
            random_state=0, # done
            ):
        """

        Parameters
        ----------
        units           : int, default = 6
            the number of recurring units
        random_state    : int, default = 0
            seed to any needed random actions

        random_state    -    random_state=0,
            units=6,
            units: Positive integer, dimensionality of the output space.
        input_shape     -    input_shape=input_layer.shape,
        activation      -    activation="linear",
        use_bias=False  -    use_bias=False,
            kernel_initializer="glorot_uniform",
            recurrent_initializer="orthogonal",
            bias_initializer="zeros",
        dropout=0.0     -    dropout=0.0,
            recurrent_dropout=0.0,
            """
        super().__init__()

        # define the architecture of the RNN here
        self.model = nn.RNN(input_size=50, hidden_size=64, batch_first=True)
        # available parameters
        # input_size, hidden_size, num_layers=1, nonlinearity='tanh', 
        # bias=True, batch_first=False, dropout=0.0, bidirectional=False, 
        # device=None, dtype=None
        
        # output layer
        self.fullyConnectedLayer = nn.Linear(in_features=64, out_features=vocab_size)


    def forward(self, X):
        hidden, final = self.model(X)
        output = self.fullyConnectedLayer(final.squeeze(0))
        return output
        
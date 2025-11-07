"""Time-CNN network for classification & regression in PyTorch."""

__authors__ = ["RecreationalMath"]
__all__ = ["CNNNetworkTorch"]

import numpy as np

from sktime.utils.dependencies import _safe_import

# handling soft dependencies for Torch modules
NNModule = _safe_import("torch.nn.Module")


class CNNNetworkTorch(NNModule):
    """Establish the network structure for a Time-CNN in PyTorch.

    Adapted from the implementation used in [1]_.

    Parameters
    ----------
    input_shape : tuple of int of size 3
        shape of the input data fed into the network
    num_classes : int
        Number of classes to predict, 1 for regression
    n_conv_layers : int, default = 2
        Number of convolutional layers.
    out_channels : int or list of int, default = 64
        Number of output channels in the convolutional layers.
        If an int is provided, then all convolutional layers will have the same number
        of output channels. If a list is provided, then it must have length equal to
        n_conv_layers, and each element specifies the number of output channels in the
        corresponding layer. This parameter is equivalent to 'filters' in Keras.
    conv_kernel_size : int or tuple or list of int, default = 7
        Size of the convolving kernel. If an int is provided, then all convolutional
        layers will have the same kernel size. If a list is provided, then it must
        have length equal to n_conv_layers, and each element specifies the kernel size
        in the corresponding layer.
    conv_padding : str or int or a tuple of int, default = "auto"
        Controls padding logic for the convolutional layers,
        i.e. whether ``'valid'`` and ``'same'`` are passed to the ``Conv1D`` layer.
        - "auto": as per original implementation, ``"same"`` is passed if
          ``input_shape[0] < 60`` in the input layer, and ``"valid"`` otherwise.
        - "valid", "same", and other values are passed directly to ``Conv1D``
        - if a tuple of int is provided, it must specify the amount of implicit
        padding applied on both sides of the input.
    avg_pool_kernel_size : int or list of int, default = 3
        Size of the average pooling kernel.
        If an int is provided, then all average pooling layers will have the same
        kernel size. If a list is provided, then it must have length equal to
        n_conv_layers, and each element specifies the kernel size in the corresponding
        average pooling layer.
    activation_hidden : str or None or an instance of activation functions defined in
        torch.nn, default = "relu"
        The activation function applied inside the convolutional layers of the CNN.
        Can be any of "relu", "leakyrelu", "elu", "prelu", "gelu", "selu",
        "rrelu", "celu", "tanh", "hardtanh".
    activation : str or None or an instance of activation functions defined in
        torch.nn, default = None
        Activation function used in the fully connected output layer. List of supported
        activation functions: ['sigmoid', 'softmax', 'logsoftmax', 'logsigmoid'].
        If None, then no activation function is applied.
    bias : bool, default = True
        If False, then the layer does not use bias weights.
    fc_dropout : float, default = 0.0
        If non-zero, introduces a Dropout layer on the outputs of the fully connected
        layers, with dropout probability equal to fc_dropout.
    random_state   : int, default = 0
        Seed to ensure reproducibility.

    References
    ----------
    .. [1] Zhao et al. Convolutional neural networks for time series classification,
    Journal of Systems Engineering and Electronics 28(1), 162--169, 2017
    .. [2] Fawaz et al. Deep learning for time series classification: a review,
    Data Mining and Knowledge Discovery 33, 917--963, 2019
    """

    _tags = {
        "authors": ["RecreationalMath"],
        "maintainers": ["RecreationalMath"],
        "python_dependencies": ["torch"],
        "capability:random_state": True,
        "property:randomness": "stochastic",
    }

    def __init__(
        self: "CNNNetworkTorch",
        input_shape: tuple[int, int, int],
        num_classes: int,
        n_conv_layers: int = 2,
        out_channels: int | list[int] = 64,
        conv_kernel_size: int | list[int] = 7,
        conv_padding: str | int | tuple[int, int] = "auto",
        avg_pool_kernel_size: int | list[int] = 3,
        activation_hidden: str | None | callable = "relu",
        activation: str | None | callable = None,
        bias: bool = True,
        fc_dropout: float = 0.0,
        random_state: int = 0,
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        # convolutional-layer-specific parameters
        self.n_conv_layers = n_conv_layers
        self.out_channels = out_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_padding = conv_padding
        self.bias = bias
        self.activation_hidden = activation_hidden
        # average-pooling-layer-specific parameters
        self.avg_pool_kernel_size = avg_pool_kernel_size
        # fully-connected-layer-specific parameters
        self.activation = activation
        self.fc_dropout = fc_dropout
        self.random_state = random_state

        super().__init__()

        # validating the input_shape
        self._validate_input_shape()
        # extracting num_channels/num_features from input_shape
        _in_channels = self.input_shape[1]  # n_dims

        # validating convolutional-layer parameters
        self._validate_out_channels()
        self._validate_conv_kernel_size()
        # instantiating the padding parameter to be used in convolutional layers
        # after this call, self._conv_padding will be available
        self._instantiate_conv_padding()
        # validating pooling-layer parameters
        self._validate_avg_pool_kernel_size()

        # defining the model architecture
        layers = []

        # importing required torch modules
        nnConv1d = _safe_import("torch.nn.Conv1d")
        nnAvgPool1d = _safe_import("torch.nn.AvgPool1d")

        for i in range(self.n_conv_layers):
            # determining out_channels for the current layer
            if isinstance(self.out_channels, int):
                _out_channels = self.out_channels
            else:
                _out_channels = self.out_channels[i]

            # determining conv_kernel_size for the current convolutional layer
            if isinstance(self.conv_kernel_size, int):
                _conv_kernel_size = self.conv_kernel_size
            else:
                _conv_kernel_size = self.conv_kernel_size[i]

            # adding convolutional layer
            layers.append(
                nnConv1d(
                    in_channels=_in_channels,
                    out_channels=_out_channels,
                    kernel_size=_conv_kernel_size,
                    padding=self._conv_padding,
                    bias=self.bias,
                )
            )

            # adding activation function after convolutional layer
            if self.activation_hidden is not None:
                layers.append(self._get_activation_function(layer="hidden"))

            # determining avg_pool_kernel_size for the current layer
            if isinstance(self.avg_pool_kernel_size, int):
                _avg_pool_kernel_size = self.avg_pool_kernel_size
            else:
                _avg_pool_kernel_size = self.avg_pool_kernel_size[i]

            # adding average pooling layer
            layers.append(nnAvgPool1d(kernel_size=_avg_pool_kernel_size))
            # updating in_channels for the next layer
            _in_channels = _out_channels

        # flattening the output from the convolutional layers
        nnFlatten = _safe_import("torch.nn.Flatten")
        layers.append(nnFlatten())

        # defining the model
        nnSequential = _safe_import("torch.nn.Sequential")
        self.model = nnSequential(*layers)

        # defining the fully connected output layer
        fc = []
        nnDropout = _safe_import("torch.nn.Dropout")
        nnLinear = _safe_import("torch.nn.Linear")
        if self.fc_dropout:
            fc.append(nnDropout(p=self.fc_dropout))
        fc.append(
            nnLinear(
                in_features=_in_channels,
                out_features=self.num_classes,
                bias=self.bias,
            )
        )
        if self.activation:
            fc.append(self._get_activation_function(layer="output"))
        self.output_layer = nnSequential(*fc)

    def forward(self, X):
        """Forward pass through the network.

        Parameters
        ----------
        X : torch.Tensor of shape (seq_length, batch_size input_size)
            Input tensor containing the time series data.

        Returns
        -------
        out : torch.Tensor of shape (seq_length, batch_size, hidden_size)
            Output tensor containing the hidden states for each time step.
        """
        if isinstance(X, np.ndarray):
            torchFrom_numpy = _safe_import("torch.from_numpy")
            X = torchFrom_numpy(X).float()
            # X = X.permute(1, 0, 2)
            # X = X.unsqueeze(0)

        out = self.model(X)
        out = self.output_layer(out)
        return out

    def _validate_input_shape(self):
        """Validate the input shape parameter.

        3D input is expected: (n_instances, n_dims, series_length)

        Raises
        ------
        TypeError
            If input_shape is not a tuple.
        ValueError
            If input_shape is not of size 3.
        """
        if not isinstance(self.input_shape, tuple):
            raise TypeError(
                "`input_shape` should be of type tuple. "
                f"But found the type to be: {type(self.input_shape)}"
            )
        if len(self.input_shape) != 3:
            raise ValueError(
                "`input_shape` should be a tuple of size 3. "
                f"But found the size to be: {len(self.input_shape)}"
            )

    def _validate_out_channels(self):
        """Validate the out_channels parameter.

        If out_channels is passed as a list,
        its length must equal n_conv_layers and all elements must be of type int.

        Raises
        ------
        TypeError
            If out_channels is a list but not all elements are of type int.
        ValueError
            If out_channels is a list but its length does not equal n_conv_layers.
        """
        if isinstance(self.out_channels, list):
            if len(self.out_channels) != self.n_conv_layers:
                raise ValueError(
                    "If `out_channels` is passed as a list. Its length must be "
                    "equal to `n_conv_layers`. Found length of "
                    f"`out_channels` as {len(self.out_channels)}"
                    f" and `n_conv_layers` as {self.n_conv_layers}"
                )
            if not all(isinstance(x, int) for x in self.out_channels):
                raise TypeError(
                    "If `out_channels` is passed as a list. "
                    "All elements in the `out_channels` list must be of type int. "
                    "Please check the values provided."
                )

    def _validate_conv_kernel_size(self):
        """Validate the conv_kernel_size parameter.

        if conv_kernel_size is passed a list its length must equal n_conv_layers.

        Raises
        ------
        TypeError
            If conv_kernel_size is a list but not all elements are of type int or tuple.
        ValueError
            If conv_kernel_size is a list but its length does not equal n_conv_layers.
        """
        if isinstance(self.conv_kernel_size, list):
            if len(self.conv_kernel_size) != self.n_conv_layers:
                raise ValueError(
                    "If `conv_kernel_size` is passed as a list. Its length must be "
                    "equal to `n_conv_layers`. Found length of "
                    f"`conv_kernel_size` as {len(self.conv_kernel_size)}"
                    f" and `n_conv_layers` as {self.n_conv_layers}"
                )
            if not all(isinstance(x, (int, tuple)) for x in self.conv_kernel_size):
                raise TypeError(
                    "If `conv_kernel_size` is passed as a list. "
                    "All elements in the `conv_kernel_size` list must be of type int . "
                    "or tuple. Please check the values provided."
                )

    def _instantiate_conv_padding(self):
        """Instantiate the conv_padding parameter.

        Instantiate conv_padding based on the value provided during initialization
        and if value is "auto", use strategy defined in the [1]_.

        References
        ----------
        .. [1] Zhao et al. Convolutional neural networks for time series classification,
        Journal of Systems Engineering and Electronics 28(1), 162--169, 2017
        """
        if self.conv_padding == "auto":
            if self.input_shape[0] < 60:
                self._conv_padding = "same"
            else:
                self._conv_padding = "valid"
        else:
            self._conv_padding = self.conv_padding

    def _validate_avg_pool_kernel_size(self):
        """Validate the avg_pool_kernel_size parameter.

        If avg_pool_kernel_size is passed as a list its length must equal n_conv_layers.

        Raises
        ------
        TypeError
            If avg_pool_kernel_size is a list
            but not all elements are of type int or tuple.
        ValueError
            If avg_pool_kernel_size is a list but its length does not equal
            n_conv_layers.
        """
        if isinstance(self.avg_pool_kernel_size, list):
            if len(self.avg_pool_kernel_size) != self.n_conv_layers:
                raise ValueError(
                    "If `avg_pool_kernel_size` is passed as a list. Its length must "
                    "be equal to `n_conv_layers`. Found length of "
                    f"`avg_pool_kernel_size` as {len(self.avg_pool_kernel_size)}"
                    f" and `n_conv_layers` as {self.n_conv_layers}"
                )
            if not all(isinstance(x, (int, tuple)) for x in self.avg_pool_kernel_size):
                raise TypeError(
                    "If `avg_pool_kernel_size` is passed as a list. "
                    "All elements in the `avg_pool_kernel_size` list must be of type "
                    "int or tuple. Please check the values provided."
                )

    def _get_activation_function(self, layer: str):
        """Instantiate and return the activation function.

        Parameters
        ----------
        layer : str
            Specifies whether the activation function is to be used in the hidden
            convolutional layers or in the output fully connected layer.
            Accepted values are "hidden" and "output".

        Returns
        -------
        activation_function : callable (torch.nn.Module)
            An instance of the specified activation function.

        Raises
        ------
        ValueError
            If an unsupported activation function is specified.
        """
        if layer == "hidden":
            if isinstance(self.activation_hidden, NNModule):
                return self.activation_hidden
            elif isinstance(self.activation_hidden, str):
                if self.activation_hidden.lower() == "relu":
                    return _safe_import("torch.nn.ReLU")()
                elif self.activation_hidden.lower() == "leakyrelu":
                    return _safe_import("torch.nn.LeakyReLU")()
                elif self.activation_hidden.lower() == "elu":
                    return _safe_import("torch.nn.ELU")()
                elif self.activation_hidden.lower() == "prelu":
                    return _safe_import("torch.nn.PReLU")()
                elif self.activation_hidden.lower() == "gelu":
                    return _safe_import("torch.nn.GELU")()
                elif self.activation_hidden.lower() == "selu":
                    return _safe_import("torch.nn.SELU")()
                elif self.activation_hidden.lower() == "rrelu":
                    return _safe_import("torch.nn.RReLU")()
                elif self.activation_hidden.lower() == "celu":
                    return _safe_import("torch.nn.CELU")()
                elif self.activation_hidden.lower() == "tanh":
                    return _safe_import("torch.nn.Tanh")()
                elif self.activation_hidden.lower() == "hardtanh":
                    return _safe_import("torch.nn.Hardtanh")()
                else:
                    raise ValueError(
                        "If `activation_hidden` is not None, it must be one of "
                        "'relu', 'leakyrelu', 'elu', 'prelu', 'gelu', 'selu', "
                        "'rrelu', 'celu', 'tanh', 'hardtanh'. "
                        f"But found {self.activation_hidden}"
                    )
            else:
                raise TypeError(
                    "`activation_hidden` should either be of type torch.nn.Module or"
                    f" str. But found the type to be: {type(self.activation_hidden)}"
                )
        elif layer == "output":
            if isinstance(self.activation, NNModule):
                return self.activation
            elif isinstance(self.activation, str):
                if self.activation.lower() == "sigmoid":
                    return _safe_import("torch.nn.Sigmoid")()
                elif self.activation.lower() == "softmax":
                    return _safe_import("torch.nn.Softmax")(dim=1)
                elif self.activation.lower() == "logsoftmax":
                    return _safe_import("torch.nn.LogSoftmax")(dim=1)
                elif self.activation.lower() == "logsigmoid":
                    return _safe_import("torch.nn.LogSigmoid")()
                else:
                    raise ValueError(
                        "If `activation` is not None, it must be one of "
                        "'sigmoid', 'logsigmoid', 'softmax' or 'logsoftmax'. "
                        f"Found {self.activation}"
                    )
            else:
                raise TypeError(
                    "`activation` should either be of type str or torch.nn.Module. "
                    f"But found the type to be: {type(self.activation)}"
                )

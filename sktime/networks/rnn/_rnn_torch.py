"""Time Recurrent Neural Network (RNN) (minus the final output layer)."""

__authors__ = ["RecreationalMath"]

from sktime.networks.base import BaseDeepNetwork


class RNNNetwork(BaseDeepNetwork):
    """Establish the network structure for an RNN.

     Parameters
    ----------
    units           : int, default = 6
        the number of recurring units
    random_state    : int, default = 0
        seed to any needed random actions
    """

    def __init__(
        self,
        units=6,
        random_state=0,
    ):
        self.random_state = random_state
        self.units = units
        super().__init__()

    def build_network(self, input_shape, **kwargs):
        """Construct a compiled, un-trained, keras model that is ready for training.

        Parameters
        ----------
        input_shape : int or tuple
            The shape of the data fed into the input layer. It should either
            have dimensions of (m, d) or m. In case an int is passed,
            1 is appended for d.

        Returns
        -------
        output : a compiled Keras Model
        """
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense


class MLP(tf.keras.Model):
    """
    A Multilayer Perceptron (MLP) model.
    """

    def __init__(self, layers: list[int], activations: list[str]) -> None:
        """
        Initializes the MLP with the given configuration.

        :param layers: Layer sizes of dense layers
        :param activations: Activation functions for respective layers
        """
        super(MLP, self).__init__()
        assert len(layers) == len(activations), "Number of layers does not match number of activation functions"
        assert len(layers) >= 2, "At least input and ouput layer must be provided"

        self.model = Sequential()
        for input_dim, output_dim, activation in zip(layers[0:-1], layers[1:], activations):
            self.model.add(
                Dense(output_dim, input_dim=input_dim, activation=activation))

    def call(self, inputs):
        return self.model(inputs)

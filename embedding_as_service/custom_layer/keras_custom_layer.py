from keras import backend as K
from keras.layers import Layer


class DenseLayer(Layer):
    def __init__(self, output_dim=64, activation='relu', **kwargs):
        self.output_dim = output_dim,
        self.activation = activation
        super(DenseLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.dense = Layer.Dense(units=self.output_dim,
                                 activation=self.activation,
                                 trainable=True)

        super(DenseLayer, self).build(input_shape)

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
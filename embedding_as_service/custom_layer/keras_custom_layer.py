from keras import backend as K
from keras.layers import Layer


class DenseLayer(Layer):
    def __init__(self, input_shape, activation, output_dim, **kwargs):
        self.input_shape = input_shape
        self.activation = activation
        self.output_dim = output_dim
        super(DenseLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.dense = Layer.Dense(units=self.output_dim,
                                 activation=self.activation,
                                 trainable=True)

        super(DenseLayer, self).build(input_shape)

    def call(self, x):
        return K.dot(x, self.dense)

    def compute_output_shape(self):
        return self.output_dim
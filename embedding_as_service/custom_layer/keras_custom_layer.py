from keras import backend as K
from keras.layers import Layer


class DenseLayer(Layer):
    def __init__(self, input_shape, output_dim, activation, **kwargs):
        self.input_shape = input_shape,
        self.output_dim = output_dim,
        self.activation = activation
        super(DenseLayer, self).__init__(**kwargs)

    def build(self):

        self.dense = Layer.Dense(units=self.output_dim,
                                 activation=self.activation,
                                 trainable=True)

        super(DenseLayer, self).build()

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (self.input_shape[0], self.output_dim)
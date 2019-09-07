from keras import backend as K
from keras.layers import Layer


class DenseLayer(Layer):
    def __init__(self, **kwargs):

        super(DenseLayer, self).__init__(**kwargs)

    def build(self, input_shape,output_dim, activation):

        self.dense = Layer.Dense(units=self.output_dim,
                                 activation=self.activation,
                                 trainable=True)

        super(DenseLayer, self).build(input_shape, output_dim, activation)

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
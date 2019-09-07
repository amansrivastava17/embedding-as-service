from keras import backend as K
from keras.layers import Layer


class DenseLayer(Layer):
    def __init__(self, **kwargs):
        super(DenseLayer, self).__init__(**kwargs)

    def build(self, input_shape, output_shape, activation):

        self.dense = Layer.Dense(units=output_shape,
                                 activation=activation,
                                 trainable=True)

        super(DenseLayer, self).build(input_shape, output_shape, activation)

    def call(self, x):
        return K.dot(x, self.dense)

    def compute_output_shape(self, output_shape):
        return self.output_shape
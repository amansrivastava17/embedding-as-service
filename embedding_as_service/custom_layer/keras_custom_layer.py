from keras import backend as K
from keras.layers import Layer


class DenseLayer(Layer):
    def __init__(self, output_dimension=64, activation_func='relu', **kwargs):
        self.output_dimension = output_dimension
        self.activation_func = activation_func
        super(DenseLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.dense = Layer.Dense(self.output_dimension,
                                 activation=self.activation_func,
                                 trainable=True)

        super(DenseLayer, self).build(input_shape)

    def call(self, x):
        return K.dot(x, self.dense)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
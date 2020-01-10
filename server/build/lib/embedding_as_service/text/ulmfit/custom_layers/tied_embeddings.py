from __future__ import absolute_import

from keras import backend as K
from keras import activations
from keras.layers import Layer


class TiedEmbeddingsTransposed(Layer):
    """Layer for tying embeddings in an output layer.
    A regular embedding layer has the shape: V x H (V: size of the vocabulary. H: size of the projected space).
    In this layer, we'll go: H x V.
    With the same weights than the regular embedding.
    In addition, it may have an activation.
    # References
        - [ Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859)
    """

    def __init__(self, tied_to=None,
                 activation=None,
                 **kwargs):
        super(TiedEmbeddingsTransposed, self).__init__(**kwargs)
        self.tied_to = tied_to
        self.activation = activations.get(activation)

    def build(self, input_shape):
        self.transposed_weights = K.transpose(self.tied_to.weights[0])
        self.built = True

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape[0], K.int_shape(self.tied_to.weights[0])[0]

    def call(self, inputs, mask=None):
        output = K.dot(inputs, self.transposed_weights)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = {'activation': activations.serialize(self.activation)
                  }
        base_config = super(TiedEmbeddingsTransposed, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

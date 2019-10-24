import tensorflow as tf
import tensorflow_hub as hub
import os
import re
import numpy as np
import random
from tqdm import tqdm_notebook
from tensorflow.keras import backend as TK
from keras.layers import Layer


class BertLayer(Layer):
    def __init__(
            self,
            n_fine_tune_layers,
            pooling,
            intermediate_layer,
            bert_path,
            output_size,
            trainable,
            n_total_layers,
            **kwargs,
    ):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.pooling = pooling
        self.intermediate_layer = intermediate_layer
        self.bert_path = bert_path
        self.trainable = trainable
        self.output_size = output_size
        self.n_total_layers = n_total_layers
        super(BertLayer, self).__init__(**kwargs)

    def get_config(self):
        config = {'n_fine_tune_layers': self.n_fine_tune_layers, 'pooling': self.pooling,
                  'intermediate_layer': self.intermediate_layer, 'output_size': self.output_size,
                  'bert_path': self.bert_path, 'n_total_layers': self.n_total_layers}

        base_config = super(BertLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_intermediate_layer(self, last_layer, total_layers, desired_layer):
        """
        Method to get outputs from any intermediate layer of bert model.
        """
        intermediate_layer_name = last_layer.name.replace(str(total_layers + 1),
                                                          str(desired_layer + 1))
        return tf.get_default_graph().get_tensor_by_name(intermediate_layer_name)

    def build(self, input_shape):
        self.bert = hub.Module(
            self.bert_path, trainable=self.trainable, name=f"{self.name}_module"
        )

        # Remove unused layers
        trainable_vars = self.bert.variables
        if self.pooling == "first":
            trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
            trainable_layers = ["pooler/dense"]

        elif self.pooling == "mean":
            trainable_vars = [
                var
                for var in trainable_vars
                if not "/cls/" in var.name and not "/pooler/" in var.name
            ]
            trainable_layers = []
        else:
            trainable_layers = []

        # Select how many layers to fine tune
        if self.intermediate_layer:
            last_layer = self.intermediate_layer - 1
        else:
            last_layer = self.n_total_layers - 1

        assert last_layer - self.n_fine_tune_layers - 1 > 0, "Not enough layers to train"

        for i in range(self.n_fine_tune_layers):
            trainable_layers.append(f"encoder/layer_{str(last_layer - i)}")

        # Update trainable vars to contain only the specified layers
        trainable_vars = [
            var
            for var in trainable_vars
            if any([l in var.name for l in trainable_layers])
        ]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [TK.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids
        )
        mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
        if self.pooling == "first":
            pooled = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "pooled_output"
            ]
        elif self.pooling == "mean":
            result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "sequence_output"
            ]
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
            input_mask = tf.cast(input_mask, tf.float32)
            pooled = masked_reduce_mean(result, input_mask)
        elif self.pooling is None:
            result = self.bert(inputs=bert_inputs, signature="tokens",
                               as_dict=True)["sequence_output"]
            if self.intermediate_layer:
                # Getting embeddings from intermediate layer.
                result = self.get_intermediate_layer(last_layer=result,
                                                     total_layers=self.n_total_layers,
                                                     desired_layer=self.intermediate_layer)

            input_mask = tf.cast(input_mask, tf.float32)
            # Making [MASK] and padding tokens.
            result_zero_padded = mul_mask(result, input_mask)
            return result_zero_padded
        return pooled

    def compute_output_shape(self, input_shape):
        if not self.pooling:
            return input_shape[0][0], input_shape[0][1], self.output_size
        else:
            return input_shape[0], self.output_size
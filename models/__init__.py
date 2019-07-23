# coding=utf-8
from __future__ import absolute_import
from typing import NamedTuple

import tensorflow as tf
init_op = tf.global_variables_initializer()

TF_SESS = tf.Session()
TF_SESS.run(init_op)


MODELS_DIR = '.embeddings'


class Embedding(NamedTuple):
    name: str
    dimensions: int
    corpus_size: str
    vocabulary_size: str
    download_url: str
    format: str
    architecture: str
    trained_data: str
    language: str


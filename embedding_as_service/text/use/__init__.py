from typing import List, Dict, Optional, Union
import numpy as np

from embedding_as_service.text import Embedding
import tensorflow as tf
import tensorflow_hub as hub


class Embeddings(object):
    EMBEDDING_MODELS: List[Embedding] = [
                        Embedding(name=u'use_dan',
                                  dimensions=512,
                                  corpus_size='na',
                                  vocabulary_size='230k',
                                  download_url='https://storage.googleapis.com/tfhub-modules/'
                                               'google/universal-sentence-encoder/2.tar.gz',
                                  format='tar.gz',
                                  architecture='DAN',
                                  trained_data='wikipedia and other sources',
                                  language='en'),
                        Embedding(name=u'use_transformer_large',
                                  dimensions=512,
                                  corpus_size='na',
                                  vocabulary_size='230k',
                                  download_url='https://storage.googleapis.com/tfhub-modules/'
                                               'google/universal-sentence-encoder-large/3.tar.gz',
                                  format='tar.gz',
                                  architecture='Transformer',
                                  trained_data='wikipedia and other sources',
                                  language='en'),
                        Embedding(name=u'use_transformer_lite',
                                  dimensions=512,
                                  corpus_size='na',
                                  vocabulary_size='na',
                                  download_url='https://storage.googleapis.com/tfhub-modules/'
                                               'google/universal-sentence-encoder-lite/2.tar.gz',
                                  format='tar.gz',
                                  architecture='Transformer',
                                  trained_data='wikipedia and other sources',
                                  language='en')
                        ]
    EMBEDDING_MODELS: Dict[str, Embedding] = {embedding.name: embedding for embedding in EMBEDDING_MODELS}

    def __init__(self):
        self.sess = tf.Session()
        self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        self.use_module = None
        self.model_name = None
        self.max_seq_length = None

    def load_model(self, model: str, model_path: str, max_seq_length: int):
        g = tf.Graph()
        with g.as_default():
            self.use_module = hub.Module(model_path)
            init_op = tf.group([tf.global_variables_initializer()])
        g.finalize()
        self.sess = tf.Session(graph=g)
        self.sess.run(init_op)
        self.model_name = model
        self.max_seq_length = max_seq_length

    def encode(self, texts: Union[List[str], List[List[str]]],
               pooling: str,
               is_tokenized: bool = False,
               **kwargs
               ) -> Optional[np.array]:
        return self.sess.run(self.use_module(texts))

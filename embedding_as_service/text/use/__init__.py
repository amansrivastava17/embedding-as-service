from typing import List, Dict, Optional, Union
import numpy as np

from embedding_as_service.text import Embedding
import tensorflow as tf
import tensorflow_hub as hub
import sentencepiece as spm


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
        self.use_outputs = None
        self.model_name = None
        self.max_seq_length = None

        # placeholder for dan and large model
        self.sentences = None

        # sentencepiece and place holder model for lite version
        self.sp_model = spm.SentencePieceProcessor()
        self.input_placeholder = None

    def process_to_ids_in_sparse_format(self, sentences):
        # An utility method that processes sentences with the sentence piece processor
        # 'sp' and returns the results in tf.SparseTensor-similar format:
        # (values, indices, dense_shape)
        ids = [self.sp_model.EncodeAsIds(x) for x in sentences]
        max_len = max(len(x) for x in ids)
        dense_shape = (len(ids), max_len)
        values = [item for sublist in ids for item in sublist]
        indices = [[row, col] for row in range(len(ids)) for col in range(len(ids[row]))]
        return values, indices, dense_shape

    def load_model(self, model: str, model_path: str, max_seq_length: int):
        spm_path_info = None
        g = tf.Graph()
        with g.as_default():
            hub_module = hub.Module(model_path)
            if model == 'use_transformer_lite':
                self.input_placeholder = tf.sparse_placeholder(tf.int64, shape=[None, None])
                self.use_outputs = hub_module(
                    inputs=dict(
                        values=self.input_placeholder.values,
                        indices=self.input_placeholder.indices,
                        dense_shape=self.input_placeholder.dense_shape)
                )
                spm_path_info = hub_module(signature="spm_path")
            else:
                self.sentences = tf.placeholder(tf.string, shape=[None])
                self.use_outputs = hub_module(self.sentences, as_dict=True)
            init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])

        g.finalize()
        self.sess = tf.Session(graph=g)
        self.sess.run(init_op)

        if model == 'use_transformer_lite':
            spm_path = self.sess.run(spm_path_info)
            self.sp_model.Load(spm_path)

        self.model_name = model
        self.max_seq_length = max_seq_length

    def encode(self, texts: Union[List[str], List[List[str]]],
               pooling: str,
               is_tokenized: bool = False,
               **kwargs
               ) -> Optional[np.array]:
        if self.model_name == 'use_transformer_lite':
            values, indices, dense_shape = self.process_to_ids_in_sparse_format(texts)
            embeddings = self.sess.run(self.use_outputs, feed_dict={
                self.input_placeholder.values: values,
                self.input_placeholder.indices: indices,
                self.input_placeholder.dense_shape: dense_shape
            })
        else:
            embeddings = self.sess.run(self.use_outputs, feed_dict={self.sentences: texts})["default"]
        return embeddings

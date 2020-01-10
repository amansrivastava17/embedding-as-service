from typing import List, Dict, Optional, Union
import numpy as np

from embedding_as_service.text import Embedding
import tensorflow as tf
import tensorflow_hub as hub

from embedding_as_service.utils import POOL_FUNC_MAP


class Embeddings(object):
    EMBEDDING_MODELS: List[Embedding] = [
        Embedding(name=u'elmo_bi_lm',
                  dimensions=512,
                  corpus_size='1B',
                  vocabulary_size='5.5B',
                  download_url='https://storage.googleapis.com/tfhub-modules/google/elmo/2.tar.gz',
                  format='tar.gz',
                  architecture='Embedding layer,cnn_layer_with_maxpool,2 lstm layers',
                  trained_data='One Billion Word Benchmark',
                  language='en')
    ]

    EMBEDDING_MODELS: Dict[str, Embedding] = {embedding.name: embedding for embedding in EMBEDDING_MODELS}

    def __init__(self):
        self.elmo_outputs = None
        self.model_name = None
        self.max_seq_length = None
        self.sess = tf.Session()

        # placeholder
        self.tokens = None
        self.sequence_len = None

    @classmethod
    def tokenize(cls, text: str):
        return [word.strip() for word in text.lower().strip().split()]

    @classmethod
    def padded_tokens(cls, tokens: List[str], max_seq_length: int):
        padded_token = ""
        len_tokens = len(tokens)
        if len_tokens >= max_seq_length:
            return tokens[:max_seq_length]
        else:
            padded_len = max_seq_length - len_tokens
            return tokens + [padded_token] * padded_len

    def load_model(self, model: str, model_path: str, max_seq_length: int):
        g = tf.Graph()
        with g.as_default():
            hub_module = hub.Module(model_path)
            self.tokens = tf.placeholder(dtype=tf.string, shape=[None, max_seq_length])
            self.sequence_len = tf.placeholder(dtype=tf.int32, shape=[None])

            elmo_inputs = dict(
                tokens=self.tokens,
                sequence_len=self.sequence_len
            )
            self.elmo_outputs = hub_module(elmo_inputs, signature="tokens", as_dict=True)
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

        text_tokens = texts
        if not is_tokenized:
            text_tokens = [Embeddings.tokenize(text) for text in texts]
        text_tokens = [Embeddings.padded_tokens(tokens, self.max_seq_length) for tokens in text_tokens]
        seq_length = [self.max_seq_length] * len(texts)

        elmo_inputs = {
            self.tokens: np.array(text_tokens),
            self.sequence_len: np.array(seq_length)
        }

        token_embeddings = self.sess.run(self.elmo_outputs, feed_dict=elmo_inputs)["elmo"]

        if not pooling:
            return token_embeddings
        else:
            if pooling not in POOL_FUNC_MAP.keys():
                print(f"Pooling method \"{pooling}\" not implemented")
                return None
            pooling_func = POOL_FUNC_MAP[pooling]
            pooled = pooling_func(token_embeddings, axis=1)
            return pooled

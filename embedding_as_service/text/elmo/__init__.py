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
        self.elmo_module = None
        self.model_name = None
        self.sess = tf.Session()

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

    def load_model(self, model: str, model_path: str):
        self.elmo_module = hub.Module(model_path)
        self.sess.run(tf.initializers.global_variables())
        self.model_name = model

    def encode(self, texts: Union[List[str], List[List[str]]],
               pooling: str,
               max_seq_length: int,
               is_tokenized: bool = False,
               **kwargs
               ) -> Optional[np.array]:

        text_tokens = texts
        if not is_tokenized:
            text_tokens = [Embeddings.tokenize(text) for text in texts]
        if max_seq_length:
            text_tokens = [Embeddings.padded_tokens(tokens, max_seq_length) for tokens in text_tokens]
            seq_length = [max_seq_length] * len(texts)
        else:
            seq_length = [len(tokens) for tokens in text_tokens]

        sequence_output = self.elmo_module(inputs={"tokens": text_tokens, "sequence_len": seq_length},
                                           signature="tokens", as_dict=True)["elmo"]

        token_embeddings = self.sess.run(sequence_output)

        if not pooling:
            return token_embeddings
        else:
            if pooling not in POOL_FUNC_MAP.keys():
                print(f"Pooling method \"{pooling}\" not implemented")
                return None
            pooling_func = POOL_FUNC_MAP[pooling]
            pooled = pooling_func(token_embeddings, axis=1)
            return pooled

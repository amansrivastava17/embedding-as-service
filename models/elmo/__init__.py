from typing import List, Dict, Tuple, Any, Optional
import numpy as np

from models import Embedding
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm
from bert.tokenization import FullTokenizer

# tf.enable_eager_execution()
sess = tf.Session()


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

    tokenizer: FullTokenizer = None
    elmo_module = None
    model: str

    @classmethod
    def tokenize(cls, text: str):
        return [word.strip() for word in text.lower().strip().split()]

    @classmethod
    def load_model(cls, model: str, model_path: str):
        cls.elmo_module = hub.Module(model_path)
        cls.model = model

    @classmethod
    def padded_tokens(cls, tokens: List[str], max_seq_length: int):
        padded_token = ""
        len_tokens = len(tokens)
        if len_tokens >= max_seq_length:
            return tokens[:max_seq_length]
        else:
            padded_len = max_seq_length - len_tokens
            return tokens + [padded_token] * padded_len

    @classmethod
    def encode(cls, text: str, pooling: str = 'mean', **kwargs) -> Optional[np.array]:
        texts = [text]
        text_tokens = [cls.tokenize(text) for text in texts]
        max_seq_length = kwargs.get('max_seq_length')
        if max_seq_length:
            text_tokens = [cls.padded_tokens(tokens, max_seq_length) for tokens in text_tokens]

        max_seq_length = max([len(tokens) for tokens in text_tokens])
        embeddings = cls.elmo_module(inputs={"tokens": text_tokens, "sequence_len": max_seq_length},
                                     signature="tokens", as_dict=True)["elmo"]

        if not pooling:
            return embeddings

        if pooling == 'mean':
            return tf.reduce_mean(embeddings, 0)

        elif pooling == 'max':
            return tf.reduce_max(embeddings, 0)

        elif pooling == 'min':
            return tf.reduce_min(embeddings, 0)

        elif pooling == 'mean_max':
            return tf.concat(values=[tf.reduce_mean(embeddings, 0), tf.reduce_max(embeddings, 0)], axis=0)

        else:
            print(f"Pooling method \"{pooling}\" not implemented")
            return None

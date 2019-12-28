from typing import List, Dict, Any, Optional, Union

from embedding_as_service.text import Embedding
from embedding_as_service.utils import to_unicode, POOL_FUNC_MAP
from smart_open import open
from tqdm import tqdm
import os
import numpy as np
from numpy import dtype, float32 as real, fromstring


class Embeddings(object):

    EMBEDDING_MODELS: List[Embedding] = [
        Embedding(name=u'google_news_300',
                  dimensions=300,
                  corpus_size='100B',
                  vocabulary_size='3M',
                  download_url='https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz',
                  format='gz',
                  architecture='skip-gram',
                  trained_data='Google News',
                  language='en')
    ]

    EMBEDDING_MODELS: Dict[str, Embedding] = {embedding.name: embedding for embedding in EMBEDDING_MODELS}

    def __init__(self):
        self.word_vectors: Dict[Any, Any] = {}
        self.model_name = None

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        return [x.lower().strip() for x in text.split()]

    def load_model(self, model: str, model_path: str):
        try:
            encoding = 'utf-8'
            unicode_errors = 'strict'

            model_file = [f for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f))]
            f = open(os.path.join(model_path, model_file[0]), 'rb')

            header = to_unicode(f.readline(), encoding=encoding)
            vocab_size, vector_size = (int(x) for x in header.split())  # throws for invalid file format

            binary_len = dtype(real).itemsize * vector_size
            for _ in tqdm(range(vocab_size)):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        break
                    if ch == b'':
                        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                    if ch != b'\n':  # ignore newlines in front of words (some binary files have)
                        word.append(ch)
                word = to_unicode(b''.join(word), encoding=encoding, errors=unicode_errors)

                weights = fromstring(f.read(binary_len), dtype=real).astype(real)

                self.word_vectors[word] = weights
            self.model_name = model
            print("Model loaded Successfully !")
            return self
        except Exception as e:
            print('Error loading Model, ', str(e))

    def _single_encode_text(self, text: Union[str, List[str]], oov_vector: np.array, max_seq_length: int,
                            is_tokenized: bool):

        tokens = text
        if not is_tokenized:
            tokens = Embeddings.tokenize(text)
        if len(tokens) > max_seq_length:
            tokens = tokens[0: max_seq_length]
        while len(tokens) < max_seq_length:
            tokens.append('<pad>')
        return np.array([self.word_vectors.get(token, oov_vector) for token in tokens])

    def encode(self, texts: Union[List[str], List[List[str]]],
               pooling: str,
               max_seq_length: int,
               is_tokenized: bool = False,
               **kwargs
               ) -> Optional[np.array]:
        oov_vector = np.zeros(Embeddings.EMBEDDING_MODELS[self.model_name].dimensions, dtype="float32")
        token_embeddings = np.array([self._single_encode_text(text, oov_vector, max_seq_length, is_tokenized)
                                     for text in texts])

        if not pooling:
            return token_embeddings
        else:
            if pooling not in POOL_FUNC_MAP.keys():
                raise NotImplementedError(f"Pooling method \"{pooling}\" not implemented")
            pooling_func = POOL_FUNC_MAP[pooling]
            pooled = pooling_func(token_embeddings, axis=1)
            return pooled

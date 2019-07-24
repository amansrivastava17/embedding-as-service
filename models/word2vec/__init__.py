from typing import List, Dict, Set, Optional, Union, Any

from models import Embedding
from utils import tokenizer
from tqdm import tqdm
import numpy as np
import os
import gzip
import numpy as np
from numpy import zeros, dtype, float32 as REAL, fromstring
import warnings


def any2unicode(text, encoding='utf8', errors='strict'):
    """Convert `text` (bytestring in given encoding or unicode) to unicode.
    Parameters
    ----------
    text : str
        Input text.
    errors : str, optional
        Error handling behaviour if `text` is a bytestring.
    encoding : str, optional
        Encoding of `text` if it is a bytestring.
    Returns
    -------
    str
        Unicode version of `text`.
    """
    if isinstance(text, str):
        return text
    return text.decode('utf-8')


to_unicode = any2unicode

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

    word_vectors: Dict[Any, Any] = {}
    model: str

    @classmethod
    def _tokens(cls, text: str) -> List[str]:
        return tokenizer(text, cls.EMBEDDING_MODELS[cls.model].language)

    @classmethod
    def load_model(cls, model: str, model_path: str):
        try:
            limit = False
            encoding = 'utf-8'
            unicode_errors = 'strict'

            model_file = [f for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f))]
            f = open(os.path.join(model_path, model_file[0]), 'rb')

            header = to_unicode(f.readline(), encoding=encoding)
            vocab_size, vector_size = (int(x) for x in header.split())  # throws for invalid file format
            if limit:
                vocab_size = min(vocab_size, limit)

            binary_len = dtype(REAL).itemsize * vector_size
            for _ in range(vocab_size):
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

                weights = fromstring(f.read(binary_len), dtype=REAL).astype(REAL)

                cls.word_vectors[word] = weights

            print("Model loaded Successfully !")
            cls.model = model

            return cls

        except Exception as e:
            print('Error loading Model, ', str(e))

    def encode(cls, text: str, pooling: str = 'mean', **kwargs) -> np.array:
        result = np.zeros(cls.EMBEDDING_MODELS[cls.model].dimensions, dtype="float32")
        tokens = cls._tokens(text)

        vectors = np.array([cls.word_vectors[token] for token in tokens if token in cls.word_vectors.keys()])

        if pooling == 'mean':
            result = np.mean(vectors, axis=0)

        elif pooling == 'max':
            result = np.max(vectors, axis=0)

        elif pooling == 'sum':
            result = np.sum(vectors, axis=0)

        elif pooling == 'tf-idf-sum':
            if not kwargs.get('tfidf_dict'):
                print('Must provide tfidf dict')
                return result

            tfidf_dict = kwargs.get('tfidf_dict')
            weighted_vectors = np.array([tfidf_dict.get(token) * cls.word_vectors.get(token)
                                         for token in tokens if token in cls.word_vectors.keys()
                                         and token in tfidf_dict])
            result = np.mean(weighted_vectors, axis=0)
        else:
            print(f'Given pooling method "{pooling}" not implemented in "{cls.embedding}"')
        return result

from typing import List, Dict, Any

from models import Embedding
from utils import to_unicode
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

    def encode(self, texts: list, pooling: str = 'mean', **kwargs) -> np.array:
        text = texts[0]
        result = np.zeros(Embeddings.EMBEDDING_MODELS[self.model_name].dimensions, dtype="float32")
        tokens = Embeddings.tokenize(text)

        vectors = np.array([self.word_vectors[token] for token in tokens if token in self.word_vectors.keys()])

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
            weighted_vectors = np.array([tfidf_dict.get(token) * self.word_vectors.get(token)
                                         for token in tokens if token in self.word_vectors.keys()
                                         and token in tfidf_dict])
            result = np.mean(weighted_vectors, axis=0)
        else:
            print(f'Given pooling method "{pooling}" not implemented in "{self.model_name}"')
        return result

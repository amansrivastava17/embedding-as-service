from typing import List, Dict, Any, Set, Optional
import numpy as np
from tqdm import tqdm
import os

from models import Embedding
from utils import tokenizer


class Embeddings(object):

    EMBEDDING_MODELS: List[Embedding] = [
        Embedding(name=u'wiki_news_300',
                  dimensions=300,
                  corpus_size='16B',
                  vocabulary_size='1M',
                  download_url='https://dl.fbaipublicfiles.com/fasttext/vectors-english/'
                               'wiki-news-300d-1M.vec.zip',
                  format='vec',
                  architecture='CBOW',
                  trained_data='Wikipedia 2017',
                  language='en'),

        Embedding(name=u'common_crawl_300',
                  dimensions=300,
                  corpus_size='600B',
                  vocabulary_size='2M',
                  download_url='https://dl.fbaipublicfiles.com/fasttext/vectors-english/'
                               'crawl-300d-2M.vec.zip',
                  format='vec',
                  architecture='CBOW',
                  trained_data='Common Crawl (600B tokens)',
                  language='en'),
    ]

    EMBEDDING_MODELS: Dict[str, Embedding] = {embedding.name: embedding for embedding in EMBEDDING_MODELS}

    word_vectors: Dict[Any, Any] = {}
    model: str

    @classmethod
    def _tokens(cls, text):
        return tokenizer(text, cls.EMBEDDING_MODELS[cls.model].language)

    @classmethod
    def load_model(cls, model: str, model_path: str):
        try:
            if cls.EMBEDDING_MODELS[model].format == 'vec':
                f = open(os.path.join(model_path, model), 'r')
                next(f)
                for line in tqdm(f):
                    split_line = line.split()
                    word = split_line[0]
                    cls.word_vectors[word] = np.array([float(val) for val in split_line[1:]])
                print("Model loaded Successfully !")
                cls.model = model
                return cls
        except Exception as e:
            print('Error loading Model, ', str(e))
        return cls

    @classmethod
    def encode(cls, text: str, pooling: str = 'mean', tfidf_dict: Optional[Dict[str, float]] = None) -> np.array:
        result = np.zeros(cls.EMBEDDING_MODELS[cls.model].dimensions, dtype="float32")
        tokens = cls._tokens(text)
        vectors = np.array([cls.word_vectors[token] for token in tokens if token in cls.vocab])

        if pooling == 'mean':
            result = np.mean(vectors, axis=0)

        elif pooling == 'max':
            result = np.max(vectors, axis=0)

        elif pooling == 'sum':
            result = np.sum(vectors, axis=0)

        elif pooling == 'tf-idf-sum':
            if not tfidf_dict:
                print('Must provide tfidf dict')
                return result

            weighted_vectors = np.array([tfidf_dict.get(token) * cls.word_vectors.get(token)
                                         for token in tokens if token in cls.vocab and token in tfidf_dict])
            result = np.mean(weighted_vectors, axis=0)
        else:
            print(f'Given pooling method "{pooling}" not implemented')
        return result



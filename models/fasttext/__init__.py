from typing import List, Dict, Any, Set, Optional
import numpy as np

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
    vocab: Set[str] = set()
    model_name: str

    @classmethod
    def _tokens(cls, text):
        return tokenizer(text, cls.EMBEDDING_MODELS[cls.model_name].language)

    @classmethod
    def load_model(cls, model_name: str, model_path: str):
        try:
            if cls.EMBEDDING_MODELS[model_name].format == 'vec':
                f = open(model_path, 'r')
                next(f)
                for line in f:
                    word = line.split()[0]
                    embedding = np.asarray(line.split()[1:])
                    cls.word_vectors[word] = embedding
                    cls.vocab.add(word)
                print("Model loaded Successfully !")
                cls.model_name = model_name
                return cls
        except Exception as e:
            print('Error loading Model, ', str(e))
        return cls

    @classmethod
    def encode(cls, text: str, pooling: str = 'mean', tfidf_dict: Optional[Dict[str, float]] = None) -> np.array:
        result = np.zeros(cls.EMBEDDING_MODELS[cls.model_name].dimensions, dtype="float32")
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



from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
import os

from models import Embedding


class Embeddings(object):

    EMBEDDING_MODELS: List[Embedding] = [
        Embedding(name=u'wiki_news_300',
                  dimensions=300,
                  corpus_size='16B',
                  vocabulary_size='1M',
                  download_url='https://dl.fbaipublicfiles.com/fasttext/vectors-english/'
                               'wiki-news-300d-1M.vec.zip',
                  format='zip',
                  architecture='CBOW',
                  trained_data='Wikipedia 2017',
                  language='en'),

        Embedding(name=u'wiki_news_300_sub',
                  dimensions=300,
                  corpus_size='16B',
                  vocabulary_size='1M',
                  download_url='https://dl.fbaipublicfiles.com/fasttext/vectors-english/'
                               'wiki-news-300d-1M-subword.vec.zip',
                  format='zip',
                  architecture='CBOW',
                  trained_data='Wikipedia 2017',
                  language='en'),

        Embedding(name=u'common_crawl_300',
                  dimensions=300,
                  corpus_size='600B',
                  vocabulary_size='2M',
                  download_url='https://dl.fbaipublicfiles.com/fasttext/vectors-english/'
                               'crawl-300d-2M.vec.zip',
                  format='zip',
                  architecture='CBOW',
                  trained_data='Common Crawl (600B tokens)',
                  language='en'),

        Embedding(name=u'common_crawl_300_sub',
                  dimensions=300,
                  corpus_size='600B',
                  vocabulary_size='2M',
                  download_url='https://dl.fbaipublicfiles.com/fasttext/vectors-english/'
                               'crawl-300d-2M-subword.zip',
                  format='zip',
                  architecture='CBOW',
                  trained_data='Common Crawl (600B tokens)',
                  language='en'),

    ]

    EMBEDDING_MODELS: Dict[str, Embedding] = {embedding.name: embedding for embedding in EMBEDDING_MODELS}

    def __init__(self):
        self.word_vectors: Dict[Any, Any] = {}
        self.model = None

    @classmethod
    def _tokens(cls, text):
        return [x.lower().strip() for x in text.split()]

    def load_model(self, model: str, model_path: str):
        try:
            model_file = [f for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f))]
            f = open(os.path.join(model_path, model_file[0]), 'r')
            next(f)
            for line in tqdm(f):
                split_line = line.split()
                word = split_line[0]
                self.word_vectors[word] = np.array([float(val) for val in split_line[1:]])
            print("Model loaded Successfully !")
            self.model = model
            return self
        except Exception as e:
            print('Error loading Model, ', str(e))
        return self

    def encode(self, texts: list, pooling: str = 'mean', **kwargs) -> np.array:
        text = texts[0]
        result = np.zeros(Embeddings.EMBEDDING_MODELS[self.model].dimensions, dtype="float32")
        tokens = Embeddings._tokens(text)
        vectors = np.array([self.word_vectors[token] for token in tokens if token in self.word_vectors.keys()])

        if pooling == 'mean':
            result = np.mean(vectors, axis=0)

        elif pooling == 'max':
            result = np.max(vectors, axis=0)

        elif pooling == 'sum':
            result = np.sum(vectors, axis=0)

        elif pooling == 'tf-idf-sum':
            if not kwargs.get('tfidf_dict') :
                print('Must provide tfidf dict')
                return result

            tfidf_dict = kwargs.get('tfidf_dict')
            weighted_vectors = np.array([tfidf_dict.get(token) * self.word_vectors.get(token)
                                         for token in tokens if token in self.word_vectors.keys()
                                         and token in tfidf_dict])
            result = np.mean(weighted_vectors, axis=0)
        else:
            print(f'Given pooling method "{pooling}" not implemented')
        return result



from typing import List, Dict

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

    @classmethod
    def tokens(cls, text, model_name):
        return tokenizer(text, Embeddings.EMBEDDING_MODELS[model_name].language)

    @classmethod
    def load_model(cls, model_name: str, model_path: str):
        try:
            if cls.EMBEDDING_MODELS[model_name].format == 'txt':
                f = open(model_path, 'r')
                for line in f:
                    split_line = line.split()
                    word = split_line[0]
                    embedding = np.array([float(val) for val in split_line[1:]])
                    cls.word_vectors[word] = embedding
                    cls.vocab.add(word)
                print("Model loaded Successfully !")
                cls.model_name = model_name
                return cls
        except Exception as e:
            print('Error loading Model, ', str(e))
        return cls
    def encode(self):
        pass

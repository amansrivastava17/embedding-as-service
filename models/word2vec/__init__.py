from typing import List, Dict

from models import Embedding
from utils import tokenizer


class Embeddings(object):

    EMBEDDING_MODELS: List[Embedding] = [
        Embedding(name=u'google_news_300',
                  dimensions=300,
                  corpus_size='100B',
                  vocabulary_size='3M',
                  download_url='https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz',
                  format='vec',
                  architecture='skip-gram',
                  trained_data='Google News',
                  language='en')
    ]

    EMBEDDING_MODELS: Dict[str, Embedding] = {embedding.name: embedding for embedding in EMBEDDING_MODELS}

    @classmethod
    def tokens(cls, text, model_name):
        return tokenizer(text, Embeddings.EMBEDDING_MODELS[model_name].language)

    def encode(self):
        pass

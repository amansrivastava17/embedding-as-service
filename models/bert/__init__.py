from typing import List, Dict
import numpy as np

from models import Embedding


class Embeddings(object):

    EMBEDDING_MODELS: List[Embedding] = [
                        Embedding(name=u'bert_base_uncased',
                                  dimensions=768,
                                  corpus_size='3300M',
                                  vocabulary_size='30522(sub-word)',
                                  download_url='https://storage.googleapis.com/tfhub-modules/'
                                               'google/bert_uncased_L-12_H-768_A-12/1.tar.gz',
                                  format='tar.gz',
                                  architecture='Layers=12, Hidden = 768, heads = 12',
                                  trained_data='BooksCorpus(800M) English Wikipedia (2500M) words',
                                  language='en'),

                        Embedding(name=u'bert_base_cased',
                                  dimensions=768,
                                  corpus_size='3300M',
                                  vocabulary_size='30522(sub-word)',
                                  download_url='https://storage.googleapis.com/tfhub-modules/google/'
                                               'bert_cased_L-12_H-768_A-12/1.tar.gz',
                                  format='tar.gz',
                                  architecture='Layers=12, Hidden = 768, heads = 12',
                                  trained_data='BooksCorpus(800M) English Wikipedia (2500M) words',
                                  language='en'),

                        Embedding(name=u'bert_multi_cased',
                                  dimensions=768,
                                  corpus_size='3300M',
                                  vocabulary_size='30522 (sub-word)',
                                  download_url='https://storage.googleapis.com/tfhub-modules/google/'
                                               'bert_multi_cased_L-12_H-768_A-12/1.tar.gz',
                                  format='tar.gz',
                                  architecture='Layers=12, Hidden = 768, heads = 12',
                                  trained_data='BooksCorpus(800M) English Wikipedia (2500M) words',
                                  language='en'),

                        Embedding(name=u'bert_large_uncased',
                                  dimensions=1024,
                                  corpus_size='3300M',
                                  vocabulary_size='30522 (sub-word)',
                                  download_url='https://storage.googleapis.com/tfhub-modules/google/'
                                               'bert_uncased_L-24_H-1024_A-16/1.tar.gz',
                                  format='tar.gz',
                                  architecture='Layers=24, Hidden = 1024, heads = 16',
                                  trained_data='BooksCorpus(800M) English Wikipedia (2500M) words',
                                  language='en'),

                        Embedding(name=u'bert_large_uncased',
                                  dimensions=1024,
                                  corpus_size='3300M',
                                  vocabulary_size='30522 (sub-word)',
                                  download_url='https://storage.googleapis.com/tfhub-modules/google/'
                                               'bert_uncased_L-24_H-1024_A-16/1.tar.gz',
                                  format='tar.gz',
                                  architecture='Layers=24, Hidden = 1024, heads = 16',
                                  trained_data='BooksCorpus(800M) English Wikipedia (2500M) words',
                                  language='en')
                        ]

    EMBEDDING_MODELS: Dict[str, Embedding] = {embedding.name: embedding for embedding in EMBEDDING_MODELS}

    @classmethod
    def _tokens(cls, text: str) -> List[str]:
        pass

    @classmethod
    def load_model(cls, model_name: str, model_path: str):
        pass

    @classmethod
    def encode(cls, text: str, pooling: str = 'mean') -> np.array:
        pass

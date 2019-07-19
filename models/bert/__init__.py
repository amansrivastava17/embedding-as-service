from typing import List, Dict

from models import Embedding

#####################
# bert uses wordpiece tokenizer with 30522 sub-word vocabsize
#####################


EMBEDDING_MODELS : List[Embedding] = [
                    Embedding(name=u'bert_base_uncased',
                              dimensions=768,
                              corpus_size='3300M',
                              vocabulary_size='30522(sub-word)',
                              download_url='https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1',
                              format='tar.gz',
                              architecture='Layers=12, Hidden = 768, heads = 12',
                              trained_data='BooksCorpus(800M) English Wikipedia (2500M) words',
                              language='en'),

                    Embedding(name=u'bert__base_cased',
                              dimensions=768,
                              corpus_size='3300M',
                              vocabulary_size='30522(sub-word)',
                              download_url='https://tfhub.dev/google/bert_cased_L-12_H-768_A-12/1',
                              format='tar.gz',
                              architecture='Layers=12, Hidden = 768, heads = 12',
                              trained_data='BooksCorpus(800M) English Wikipedia (2500M) words',
                              language='en'),
                    Embedding(name=u'bert_multi_cased',
                              dimensions=768,
                              corpus_size='3300M',
                              vocabulary_size='30522 (sub-word)',
                              download_url='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                              format='tar.gz',
                              architecture='Layers=12, Hidden = 768, heads = 12',
                              trained_data='BooksCorpus(800M) English Wikipedia (2500M) words',
                              language='en'),

                    Embedding(name=u'bert_large_uncased',
                              dimensions=1024,
                              corpus_size='3300M',
                              vocabulary_size='30522 (sub-word)',
                              download_url='https://tfhub.dev/google/bert_uncased_L-24_H-1024_A-16/1',
                              format='tar.gz',
                              architecture='Layers=24, Hidden = 1024, heads = 16',
                              trained_data='BooksCorpus(800M) English Wikipedia (2500M) words',
                              language='en'),
                    Embedding(name=u'bert_large_uncased',
                              dimensions=1024,
                              corpus_size='3300M',
                              vocabulary_size='30522 (sub-word)',
                              download_url='https://tfhub.dev/google/bert_uncased_L-24_H-1024_A-16/1',
                              format='tar.gz',
                              architecture='Layers=24, Hidden = 1024, heads = 16',
                              trained_data='BooksCorpus(800M) English Wikipedia (2500M) words',
                              language='en')
                                ]

EMBEDDING_MODELS: Dict[str, Embedding] = {embedding.name: embedding for embedding in EMBEDDING_MODELS}
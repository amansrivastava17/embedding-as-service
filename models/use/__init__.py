from typing import List, Dict

from models import Embedding

EMBEDDING_MODELS: List[Embedding] = [
                    Embedding(name=u'use',
                              dimensions=512,
                              corpus_size='na',
                              vocabulary_size='230k',
                              download_url='https://tfhub.dev/google/universal-sentence-encoder/2',
                              format='.tar.gz',
                              architecture='DAN',
                              trained_data='wikipedia and other sources',
                              language='en'),
                    Embedding(name=u'use_large',
                              dimensions=512,
                              corpus_size='na',
                              vocabulary_size='230k',
                              download_url='https://tfhub.dev/google/universal-sentence-encoder-large/3',
                              format='.tar.gz',
                              architecture='Transformer',
                              trained_data='wikipedia and other sources',
                              language='en'),
                    Embedding(name=u'use_lite',
                              dimensions=512,
                              corpus_size='na',
                              vocabulary_size='na',
                              download_url='https://tfhub.dev/google/universal-sentence-encoder-lite/2',
                              format='.tar.gz',
                              architecture='Transformer',
                              trained_data='wikipedia and other sources',
                              language='en')
                                ]

EMBEDDING_MODELS: Dict[str, Embedding] = {embedding.name: embedding for embedding in EMBEDDING_MODELS}
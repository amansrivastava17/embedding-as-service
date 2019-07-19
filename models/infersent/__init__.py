from typing import List, Dict

from models import Embedding

EMBEDDING_MODELS : List[Embedding] = [
                    Embedding(name=u'infersent_glove',
                              dimensions=300,
                              corpus_size='570k human-generated English sentence pairs',
                              vocabulary_size='na',
                              download_url='https://dl.fbaipublicfiles.com/infersent/infersent1.pkl',
                              format='tar.gz',
                              architecture='cbow',
                              trained_data='SNLI dataset',
                              language='en'),
                    Embedding(name=u'infersent_fasttext',
                              dimensions=300,
                              corpus_size='570k human-generated English sentence pairs',
                              vocabulary_size='na',
                              download_url='https://dl.fbaipublicfiles.com/infersent/infersent2.pkl',
                              format='tar.gz',
                              architecture='cbow',
                              trained_data='SNLI dataset',
                              language='en')
                                ]

EMBEDDING_MODELS: Dict[str, Embedding] = {embedding.name: embedding for embedding in EMBEDDING_MODELS}
from typing import List, Dict

from models import Embedding

EMBEDDING_MODELS : List[Embedding] = [
                    Embedding(name=u'bert_base_uncased',
                              dimensions=512,
                              corpus_size='1B',
                              vocabulary_size='na',
                              download_url='https://tfhub.dev/google/elmo/2',
                              format='tar.gz',
                              architecture='embedding_layer,cnn_layer_with_maxpool,2 lstm layers',
                              trained_data='One Billion Word Benchmark')
                                ]

EMBEDDING_MODELS: Dict[str, Embedding] = {embedding.name: embedding for embedding in EMBEDDING_MODELS}
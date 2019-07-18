from typing import List, Dict

from models import Embedding

EMBEDDING_MODELS : List[Embedding] = [
                    Embedding(name=u'umlfit',
                              dimensions=300,
                              corpus_size='570k human-generated English sentence pairs',
                              vocabulary_size='230k',
                              download_url='http://files.fast.ai/models/wt103/',
                              format='.h5',
                              architecture='cbow',
                              trained_data='Stephen Merity’s Wikitext 103 dataset')
                                ]

EMBEDDING_MODELS: Dict[str, Embedding] = {embedding.name: embedding for embedding in EMBEDDING_MODELS}
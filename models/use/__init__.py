from typing import List, Dict, Tuple, Any, Optional
import numpy as np

from models import Embedding
import tensorflow as tf
import tensorflow_hub as hub


sess = tf.Session()
sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

class Embeddings(object):
    EMBEDDING_MODELS: List[Embedding] = [
                        Embedding(name=u'use_dan',
                                  dimensions=512,
                                  corpus_size='na',
                                  vocabulary_size='230k',
                                  download_url='https://storage.googleapis.com/tfhub-modules/'
                                               'google/universal-sentence-encoder/2.tar.gz',
                                  format='tar.gz',
                                  architecture='DAN',
                                  trained_data='wikipedia and other sources',
                                  language='en'),
                        Embedding(name=u'use_transformer_large',
                                  dimensions=512,
                                  corpus_size='na',
                                  vocabulary_size='230k',
                                  download_url='https://storage.googleapis.com/tfhub-modules/'
                                               'google/universal-sentence-encoder-large/3.tar.gz',
                                  format='tar.gz',
                                  architecture='Transformer',
                                  trained_data='wikipedia and other sources',
                                  language='en'),
                        Embedding(name=u'use_transformer_lite',
                                  dimensions=512,
                                  corpus_size='na',
                                  vocabulary_size='na',
                                  download_url='https://storage.googleapis.com/tfhub-modules/'
                                               'google/universal-sentence-encoder-lite/2.tar.gz',
                                  format='tar.gz',
                                  architecture='Transformer',
                                  trained_data='wikipedia and other sources',
                                  language='en')
                                    ]

    EMBEDDING_MODELS: Dict[str, Embedding] = {embedding.name: embedding for embedding in EMBEDDING_MODELS}

    use_module = None
    model: str

    @classmethod
    def load_model(cls, model: str, model_path: str):
        cls.use_module = hub.Module(model_path)
        cls.model = model

    @classmethod
    def encode(cls, text: str, pooling: str = None, **kwargs) -> Optional[np.array]:
        embeddings = cls.use_module(text)
        return embeddings

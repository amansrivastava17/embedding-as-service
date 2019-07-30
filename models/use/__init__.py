from typing import List, Dict, Optional
import numpy as np

from models import Embedding, TF_SESS

import tensorflow_hub as hub


<<<<<<< HEAD
sess = TF_SESS

=======
>>>>>>> f2af2b1cec2ca89a4e1bde9a06eb2feb0113e298
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

    def __init__(self):
        self.sess = tf.Session()
        self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        self.use_module = None
        self.model = None

    def load_model(self, model: str, model_path: str):
        self.use_module = hub.Module(model_path)
        self.model = model

<<<<<<< HEAD
    @classmethod
    def encode(cls, texts: list, pooling: str = None) -> Optional[np.array]:
        embeddings = sess.run(cls.use_module(texts))
        return embeddings
=======
    def encode(self, texts: list, pooling: str = None) -> Optional[np.array]:
        return self.use_module(texts)
>>>>>>> f2af2b1cec2ca89a4e1bde9a06eb2feb0113e298

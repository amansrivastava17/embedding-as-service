from embedding_as_service.text.ulmfit import *
from embedding_as_service.text import Embedding
from embedding_as_service.text.ulmfit.model import build_language_model
from embedding_as_service.utils import POOL_FUNC_MAP, download_from_url

from typing import List, Dict, Optional, Union
import numpy as np
import pickle
import os


class Embeddings(object):
    EMBEDDING_MODELS: List[Embedding] = [
        Embedding(name=u'ulmfit_forward',
                  dimensions=300,
                  corpus_size='570k human-generated English sentence pairs',
                  vocabulary_size='230k',
                  download_url='https://www.dropbox.com/s/chtrlru8lv0viud/ulmfit_forward.zip?dl=1',
                  format='zip',
                  architecture='LSTM',
                  trained_data='Stephen Merity’s Wikitext 103 dataset',
                  language='en'),

        Embedding(name=u'ulmfit_backward',
                  dimensions=300,
                  corpus_size='570k human-generated English sentence pairs',
                  vocabulary_size='230k',
                  download_url='https://www.dropbox.com/s/3w2h03aeac51n8b/ulmfit_backword.zip?dl=1',
                  format='zip',
                  architecture='LSTM',
                  trained_data='Stephen Merity’s Wikitext 103 dataset',
                  language='en')
    ]

    EMBEDDING_MODELS: Dict[str, Embedding] = {embedding.name: embedding for embedding in EMBEDDING_MODELS}

    def __init__(self):
        self.ulmfit_model = None
        self.model_name = None
        self.word2idx = None
        self.idx2word = None

    @classmethod
    def tokenize(cls, text: str):
        return [word.strip() for word in text.lower().strip().split()]

    def load_model(self, model: str, model_path: str):
        """
            Loads architecture and weights from saved model.
            Args:
                model: Name of the model
                model_path: directory path of saved model and architecture file.
        """

        weights_path = os.path.join(model_path,  'model.h5')
        id2word_path = os.path.join(model_path, 'itos_wt103.pkl')

        with open(id2word_path, 'rb') as f:
            idx2word = pickle.load(f)

        self.word2idx = {word: idx for idx, word in enumerate(idx2word)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}

        self.ulmfit_model = build_language_model()
        self.ulmfit_model.load_weights(weights_path)
        self.model_name = model

    def encode(self, texts: Union[List[str], List[List[str]]],
               pooling: str,
               max_seq_length: int,
               is_tokenized: bool = False,
               **kwargs
               ) -> Optional[np.array]:
        tokenized_texts = texts
        if not is_tokenized:
            tokenized_texts = [Embeddings.tokenize(text) for text in texts]
        tokenized_text_words = [[self.word2idx.get(w, self.word2idx['unk'])
                                 for w in text] for text in tokenized_texts]
        embeddings = []

        for x in tokenized_text_words:
            x = np.reshape(x, (1, len(x)))
            embeddings.append(self.ulmfit_model.predict(x)[1][0])
        if not pooling:
            return embeddings
        else:
            if pooling not in POOL_FUNC_MAP.keys():
                raise NotImplementedError(f"Pooling method \"{pooling}\" not implemented")
            pooling_func = POOL_FUNC_MAP[pooling]
            pooled = pooling_func(embeddings, axis=1)
            return pooled

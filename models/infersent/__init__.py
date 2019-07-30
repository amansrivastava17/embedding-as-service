from typing import List, Dict, Any, Set, Optional
import numpy as np

import torch

from InfersentModel import InferSent

from models import Embedding
from utils import tokenizer


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

    word_vectors: Dict[Any, Any] = {}
    vocab: Set[str] = set()
    model_name: str

    @classmethod
    def _tokens(cls, text):
        return tokenizer(text, cls.EMBEDDING_MODELS[cls.model_name].language)

    @classmethod
    def load_model(cls, model_name: str, model_path: str):
        try:
            if cls.EMBEDDING_MODELS[model_name].format == 'pkl':
                if model_name == 'infersent_glove':
                    V = 1
                else:
                    V = 2
                params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                                'pool_type': pooling, 'dpout_model': 0.30, 'version': V}
                cls.infersent = InferSent(params_model)
                cls.infersent.load_state_dict(torch.load(model_path))
                return cls
        except Exception as e:
            print('Error loading Model, ', str(e))
        return cls

    @classmethod
    def encode(cls, text: str, pooling: str = 'mean', tfidf_dict: Optional[Dict[str, float]] = None) -> np.array:
        result = np.zeros(cls.EMBEDDING_MODELS[cls.model_name].dimensions, dtype="float32")

        W2V_PATH = 'fastText/crawl-300d-2M.vec'
        cls.infersent.set_w2v_path(W2V_PATH)
        cls.infersent.build_vocab(text, tokenize=False)
        embeddings = cls.infersent.encode(text, tokenize=False)
        tokens = cls._tokens(text)
        vectors = np.array([cls.word_vectors[token] for token in tokens if token in cls.vocab])

        if pooling == 'mean':
            result = np.mean(vectors, axis=0)

        elif pooling == 'max':
            result = np.max(vectors, axis=0)

        elif pooling == 'sum':
            result = np.sum(vectors, axis=0)

        elif pooling == 'tf-idf-sum':
            if not tfidf_dict:
                print('Must provide tfidf dict')
                return result

            weighted_vectors = np.array([tfidf_dict.get(token) * cls.word_vectors.get(token)
                                         for token in tokens if token in cls.vocab and token in tfidf_dict])
            result = np.mean(weighted_vectors, axis=0)
        else:
            print(f'Given pooling method "{pooling}" not implemented')
        return result
from typing import List, Dict, Any, Optional, Union

from embedding_as_service.text import Embedding
from tqdm import tqdm
import numpy as np
import os

from embedding_as_service.utils import POOL_FUNC_MAP


class Embeddings(object):

    EMBEDDING_MODELS: List[Embedding] = [
        Embedding(name=u'twitter_100',
                  dimensions=100,
                  corpus_size='27B',
                  vocabulary_size='1.2M',
                  download_url='https://www.dropbox.com/s/q2wof83a0yq7q74/glove.twitter.27B.100d.txt.zip?dl=1',
                  format='zip',
                  architecture='glove',
                  trained_data='Twitter 2B Tweets',
                  language='en'),
        Embedding(name=u'twitter_200',
                  dimensions=200,
                  corpus_size='27B',
                  vocabulary_size='1.2M',
                  download_url='https://www.dropbox.com/s/hfw00m77ibz24y5/glove.twitter.27B.200d.txt.zip?dl=1',
                  format='zip',
                  architecture='glove',
                  trained_data='Twitter 2B Tweets',
                  language='en'),
        Embedding(name=u'twitter_25',
                  dimensions=25,
                  corpus_size='27B',
                  vocabulary_size='1.2M',
                  download_url='https://www.dropbox.com/s/jx97sz8skdp276k/glove.twitter.27B.25d.txt.zip?dl=1',
                  format='zip',
                  architecture='glove',
                  trained_data='Twitter 2B Tweets',
                  language='en'),

        Embedding(name=u'twitter_50',
                  dimensions=50,
                  corpus_size='27B',
                  vocabulary_size='1.2M',
                  download_url='https://www.dropbox.com/s/9mutj8syz3q20e3/glove.twitter.27B.50d.txt.zip?dl=1',
                  format='zip',
                  architecture='glove',
                  trained_data='Twitter 2B Tweets',
                  language='en'),
        Embedding(name=u'wiki_100',
                  dimensions=100,
                  corpus_size='6B',
                  vocabulary_size='0.4M',
                  download_url='https://www.dropbox.com/s/g0inzrsy1ds3u63/glove.6B.100d.txt.zip?dl=1',
                  format='zip',
                  architecture='glove',
                  trained_data='Wikipedia+Gigaword',
                  language='en'),
        Embedding(name=u'wiki_200',
                  dimensions=200,
                  corpus_size='6B',
                  vocabulary_size='0.4M',
                  download_url='https://www.dropbox.com/s/pmj2ycd882qkae5/glove.6B.200d.txt.zip?dl=1',
                  format='zip',
                  architecture='glove',
                  trained_data='Wikipedia+Gigaword',
                  language='en'),

        Embedding(name=u'wiki_300',
                  dimensions=300,
                  corpus_size='6B',
                  vocabulary_size='0.4M',
                  download_url='https://www.dropbox.com/s/9jbbk99p0d0n1bw/glove.6B.300d.txt.zip?dl=1',
                  format='zip',
                  architecture='glove',
                  trained_data='Wikipedia+Gigaword',
                  language='en'),

        Embedding(name=u'wiki_50',
                  dimensions=50,
                  corpus_size='6B',
                  vocabulary_size='0.4M',
                  download_url='https://www.dropbox.com/s/o3axsz1j47043si/glove.6B.50d.txt.zip?dl=1',
                  format='zip',
                  architecture='glove',
                  trained_data='Wikipedia+Gigaword',
                  language='en'),

        Embedding(name=u'crawl_42B_300',
                  dimensions=300,
                  corpus_size='42B',
                  vocabulary_size='1.9M',
                  download_url='http://nlp.stanford.edu/data/glove.42B.300d.zip',
                  format='zip',
                  architecture='glove',
                  trained_data='Common Crawl (42B tokens)',
                  language='en'),

        Embedding(name=u'crawl_840B_300',
                  dimensions=300,
                  corpus_size='840B',
                  vocabulary_size='2.2M',
                  download_url='http://nlp.stanford.edu/data/glove.840B.300d.zip',
                  format='zip',
                  architecture='glove',
                  trained_data='Common Crawl (840B tokens)',
                  language='en')

    ]

    EMBEDDING_MODELS: Dict[str, Embedding] = {embedding.name: embedding for embedding in EMBEDDING_MODELS}

    def __init__(self):
        self.word_vectors: Dict[Any, Any] = {}
        self.model_name = None

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        return [x.lower().strip() for x in text.split()]

    def load_model(self, model: str, model_path: str):
        try:
            model_file = [f for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f))]
            f = open(os.path.join(model_path, model_file[0]), 'r')
            for line in tqdm(f):
                split_line = line.split()
                word = split_line[0]
                self.word_vectors[word] = np.array([float(val) for val in split_line[1:]])
            print("Model loaded Successfully !")
            self.model_name = model
            return self
        except Exception as e:
            print('Error loading Model, ', str(e))
        return self

    def _single_encode_text(self, text: Union[str, List[str]], oov_vector: np.array, max_seq_length: int,
                            is_tokenized: bool):

        tokens = text
        if not is_tokenized:
            tokens = Embeddings.tokenize(text)
        if len(tokens) > max_seq_length:
            tokens = tokens[0: max_seq_length]
        while len(tokens) < max_seq_length:
            tokens.append('<pad>')
        return np.array([self.word_vectors.get(token, oov_vector) for token in tokens])

    def encode(self, texts: Union[List[str], List[List[str]]],
               pooling: str,
               max_seq_length: int,
               is_tokenized: bool = False,
               **kwargs
               ) -> Optional[np.array]:
        oov_vector = np.zeros(Embeddings.EMBEDDING_MODELS[self.model_name].dimensions, dtype="float32")
        token_embeddings = np.array([self._single_encode_text(text, oov_vector, max_seq_length, is_tokenized)
                                     for text in texts])

        if not pooling:
            return token_embeddings
        else:
            if pooling not in POOL_FUNC_MAP.keys():
                raise NotImplementedError(f"Pooling method \"{pooling}\" not implemented")
            pooling_func = POOL_FUNC_MAP[pooling]
            pooled = pooling_func(token_embeddings, axis=1)
            return pooled
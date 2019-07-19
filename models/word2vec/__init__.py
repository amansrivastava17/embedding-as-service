from typing import List, Dict

from models import Embedding
from utils import tokenizer


class Embeddings(object):

    EMBEDDING_MODELS: List[Embedding] = [
        Embedding(name=u'google_news_300',
                  dimensions=300,
                  corpus_size='100B',
                  vocabulary_size='3M',
                  download_url='https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz',
                  format='vec',
                  architecture='skip-gram',
                  trained_data='Google News',
                  language='en')
    ]

    EMBEDDING_MODELS: Dict[str, Embedding] = {embedding.name: embedding for embedding in EMBEDDING_MODELS}

    @classmethod
    def _tokens(cls, text: str) -> List[str]:
        return tokenizer(text, Embeddings.EMBEDDING_MODELS[model_name].language)

    @classmethod
    def load_model(cls, model_name: str, model_path: str):
        try:
            if cls.EMBEDDING_MODELS[model_name].format == '.vec':
                f = open(model_path, 'r')
                for line in f:
                    split_line = line.split()
                    word = split_line[0]
                    embedding = np.array([float(val) for val in split_line[1:]])
                    cls.word_vectors[word] = embedding
                    cls.vocab.add(word)
                print("Model loaded Successfully !")
                cls.model_name = model_name
                return cls
        except Exception as e:
            print('Error loading Model, ', str(e))
        return cls

    def encode(self):
        pass

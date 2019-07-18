from typing import List, Dict

from models import Embedding


class Embeddings(object):

    EMBEDDING_MODELS: List[Embedding] = [
        Embedding(name=u'twitter_100',
                  dimensions=100,
                  corpus_size='27B',
                  vocabulary_size='1.2M',
                  download_url='https://www.dropbox.com/s/q2wof83a0yq7q74/glove.twitter.27B.100d.txt.zip?dl=1',
                  format='txt',
                  architecture='glove',
                  trained_data='Twitter 2B Tweets',
                  language='en'),
        Embedding(name=u'twitter_200',
                  dimensions=200,
                  corpus_size='27B',
                  vocabulary_size='1.2M',
                  download_url='https://www.dropbox.com/s/hfw00m77ibz24y5/glove.twitter.27B.200d.txt.zip?dl=1',
                  format='txt',
                  architecture='glove',
                  trained_data='Twitter 2B Tweets',
                  language='en'),
        Embedding(name=u'twitter_25',
                  dimensions=25,
                  corpus_size='27B',
                  vocabulary_size='1.2M',
                  download_url='https://www.dropbox.com/s/jx97sz8skdp276k/glove.twitter.27B.25d.txt.zip?dl=1',
                  format='txt',
                  architecture='glove',
                  trained_data='Twitter 2B Tweets',
                  language='en'),

        Embedding(name=u'twitter_50',
                  dimensions=50,
                  corpus_size='27B',
                  vocabulary_size='1.2M',
                  download_url='https://www.dropbox.com/s/9mutj8syz3q20e3/glove.twitter.27B.50d.txt.zip?dl=1',
                  format='txt',
                  architecture='glove',
                  trained_data='Twitter 2B Tweets',
                  language='en'),
        Embedding(name=u'wiki_100',
                  dimensions=100,
                  corpus_size='6B',
                  vocabulary_size='0.4M',
                  download_url='https://www.dropbox.com/s/g0inzrsy1ds3u63/glove.6B.100d.txt.zip?dl=1',
                  format='txt',
                  architecture='glove',
                  trained_data='Wikipedia+Gigaword',
                  language='en'),
        Embedding(name=u'wiki_200',
                  dimensions=200,
                  corpus_size='6B',
                  vocabulary_size='0.4M',
                  download_url='https://www.dropbox.com/s/pmj2ycd882qkae5/glove.6B.200d.txt.zip?dl=1',
                  format='txt',
                  architecture='glove',
                  trained_data='Wikipedia+Gigaword',
                  language='en'),

        Embedding(name=u'wiki_300',
                  dimensions=300,
                  corpus_size='6B',
                  vocabulary_size='0.4M',
                  download_url='https://www.dropbox.com/s/9jbbk99p0d0n1bw/glove.6B.300d.txt.zip?dl=1',
                  format='txt',
                  architecture='glove',
                  trained_data='Wikipedia+Gigaword',
                  language='en'),

        Embedding(name=u'wiki_50',
                  dimensions=50,
                  corpus_size='6B',
                  vocabulary_size='0.4M',
                  download_url='https://www.dropbox.com/s/o3axsz1j47043si/glove.6B.50d.txt.zip?dl=1',
                  format='txt',
                  architecture='glove',
                  trained_data='Wikipedia+Gigaword',
                  language='en'),

        Embedding(name=u'crawl_42B_300',
                  dimensions=300,
                  corpus_size='42B',
                  vocabulary_size='1.9M',
                  download_url='http://nlp.stanford.edu/data/glove.42B.300d.zip',
                  format='txt',
                  architecture='glove',
                  trained_data='Common Crawl (42B tokens)',
                  language='en'),

        Embedding(name=u'crawl_840B_300',
                  dimensions=300,
                  corpus_size='840B',
                  vocabulary_size='2.2M',
                  download_url='http://nlp.stanford.edu/data/glove.840B.300d.zip',
                  format='txt',
                  architecture='glove',
                  trained_data='Common Crawl (840B tokens)',
                  language='en')

    ]

    EMBEDDING_MODELS: Dict[str, Embedding] = {embedding.name: embedding for embedding in EMBEDDING_MODELS}

    def tokenizer(self):
        pass

    def encode(self):
        pass

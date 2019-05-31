from models.utils import Embeddings, _Embeddings


class Word2Vec(Embeddings):
    google_news_300 = _Embeddings(name=u'word2vec-google_new-300',
                                  dimensions=300,
                                  corpus_size='100B',
                                  vocabulary_size='3M',
                                  download_url='https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz',
                                  format='.vec',
                                  architecture='skip-gram',
                                  trained_data='Google News'
                                  )

    _members = (
        google_news_300,
    )

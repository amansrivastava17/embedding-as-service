from models import Embeddings, _Embeddings


class Glove(Embeddings):
    twitter_100 = _Embeddings(name=u'twitter_100',
                              dimensions=100,
                              corpus_size='27B',
                              vocabulary_size='1.2M',
                              download_url='https://github.com/amansrivastava17/embedding_repository/raw/master/'
                                           'glove/glove.6B.100d.txt.zip',
                              format='txt',
                              architecture='glove',
                              trained_data='Twitter 2B Tweets')

    twitter_200 = _Embeddings(name=u'twitter_200',
                              dimensions=200,
                              corpus_size='27B',
                              vocabulary_size='1.2M',
                              download_url='https://github.com/amansrivastava17/embedding_repository/raw/master/glove/'
                                           'glove.6B.200d.txt.zip',
                              format='txt',
                              architecture='glove',
                              trained_data='Twitter 2B Tweets')

    twitter_25 = _Embeddings(name=u'twitter_25',
                             dimensions=25,
                             corpus_size='27B',
                             vocabulary_size='1.2M',
                             download_url='https://github.com/amansrivastava17/embedding_repository/raw/master/glove/'
                                          'glove.twitter.27B.25d.txt.zip',
                             format='txt',
                             architecture='glove',
                             trained_data='Twitter 2B Tweets')

    twitter_50 = _Embeddings(name=u'twitter_50',
                             dimensions=50,
                             corpus_size='27B',
                             vocabulary_size='1.2M',
                             download_url='https://github.com/amansrivastava17/embedding_repository/raw/master/glove/'
                                          'glove.twitter.27B.50d.txt.zip',
                             format='txt',
                             architecture='glove',
                             trained_data='Twitter 2B Tweets')

    wiki_100 = _Embeddings(name=u'wiki_100',
                           dimensions=100,
                           corpus_size='6B',
                           vocabulary_size='0.4M',
                           download_url='https://github.com/amansrivastava17/embedding_repository/raw/master/glove/'
                                        'glove.6B.100d.txt.zip',
                           format='txt',
                           architecture='glove',
                           trained_data='Wikipedia+Gigaword')

    wiki_200 = _Embeddings(name=u'wiki_200',
                           dimensions=200,
                           corpus_size='6B',
                           vocabulary_size='0.4M',
                           download_url='https://github.com/amansrivastava17/embedding_repository/raw/master/glove/'
                                        'glove.6B.200d.txt.zip',
                           format='txt',
                           architecture='glove',
                           trained_data='Wikipedia+Gigaword'
                           )

    wiki_300 = _Embeddings(name=u'wiki_300',
                           dimensions=300,
                           corpus_size='6B',
                           vocabulary_size='0.4M',
                           download_url='https://github.com/amansrivastava17/embedding_repository/raw/master/glove/'
                                        'glove.6B.300d.txt.zip',
                           format='txt',
                           architecture='glove',
                           trained_data='Wikipedia+Gigaword')

    wiki_50 = _Embeddings(name=u'wiki_50',
                          dimensions=50,
                          corpus_size='6B',
                          vocabulary_size='0.4M',
                          download_url='https://github.com/amansrivastava17/embedding_repository/raw/master/glove/'
                                       'glove.6B.50d.txt.zip',
                          format='txt',
                          architecture='glove',
                          trained_data='Wikipedia+Gigaword')

    crawl_42B_300 = _Embeddings(name=u'crawl_42B_300',
                                dimensions=300,
                                corpus_size='42B',
                                vocabulary_size='1.9M',
                                download_url='http://nlp.stanford.edu/data/glove.42B.300d.zip',
                                format='txt',
                                architecture='glove',
                                trained_data='Common Crawl (42B tokens)')

    crawl_840B_300 = _Embeddings(name=u'crawl_840B_300',
                                 dimensions=300,
                                 corpus_size='840B',
                                 vocabulary_size='2.2M',
                                 download_url='http://nlp.stanford.edu/data/glove.840B.300d.zip',
                                 format='txt',
                                 architecture='glove',
                                 trained_data='Common Crawl (840B tokens)')

    _members = (
        twitter_100,
        twitter_200,
        twitter_25,
        twitter_50,
        wiki_100,
        wiki_200,
        wiki_300,
        wiki_50,
        crawl_42B_300,
        crawl_840B_300
    )

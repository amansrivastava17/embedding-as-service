from models.utils import Embeddings, _Embeddings


class Glove(Embeddings):
    twitter_100 = _Embeddings(name=u'glove-twitter-100',
                              dimensions=100,
                              corpus_size='27B',
                              vocabulary_size='1.2M',
                              download_url='',
                              format='.txt',
                              architecture='glove',
                              trained_data='Twitter 2B Tweets')

    twitter_200 = _Embeddings(name=u'glove-twitter-200',
                              dimensions=200,
                              corpus_size='27B',
                              vocabulary_size='1.2M',
                              download_url='',
                              format='.txt',
                              architecture='glove',
                              trained_data='Twitter 2B Tweets')

    twitter_25 = _Embeddings(name=u'glove-twitter-25',
                             dimensions=25,
                             corpus_size='27B',
                             vocabulary_size='1.2M',
                             download_url='',
                             format='.txt',
                             architecture='glove',
                             trained_data='Twitter 2B Tweets')

    twitter_50 = _Embeddings(name=u'glove-twitter-50',
                             dimensions=50,
                             corpus_size='27B',
                             vocabulary_size='1.2M',
                             download_url='',
                             format='.txt',
                             architecture='glove',
                             trained_data='Twitter 2B Tweets')

    wiki_100 = _Embeddings(name=u'glove-wiki-gigaword-100',
                           dimensions=100,
                           corpus_size='6B',
                           vocabulary_size='0.4M',
                           download_url='',
                           format='.txt',
                           architecture='glove',
                           trained_data='Wikipedia+Gigaword')

    wiki_200 = _Embeddings(name=u'glove-wiki-gigaword-200',
                           dimensions=200,
                           corpus_size='6B',
                           vocabulary_size='0.4M',
                           download_url='',
                           format='.txt',
                           architecture='glove',
                           trained_data='Wikipedia+Gigaword'
                           )

    wiki_300 = _Embeddings(name=u'glove-wiki-gigaword-300',
                           dimensions=300,
                           corpus_size='6B',
                           vocabulary_size='0.4M',
                           download_url='',
                           format='.txt',
                           architecture='glove',
                           trained_data='Wikipedia+Gigaword')

    wiki_50 = _Embeddings(name=u'glove-wiki-gigaword-50',
                          dimensions=50,
                          corpus_size='6B',
                          vocabulary_size='0.4M',
                          download_url='',
                          format='.txt',
                          architecture='glove',
                          trained_data='Wikipedia+Gigaword')

    crawl_42B_300 = _Embeddings(name=u'glove-common-crawl-42B-300',
                                dimensions=300,
                                corpus_size='42B',
                                vocabulary_size='1.9M',
                                download_url='',
                                format='.txt',
                                architecture='glove',
                                trained_data='Common Crawl (42B tokens)')

    crawl_840B_300 = _Embeddings(name=u'glove-common-crawl-840B-300',
                                 dimensions=300,
                                 corpus_size='840B',
                                 vocabulary_size='2.2M',
                                 download_url='',
                                 format='.txt',
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

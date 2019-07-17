from models import Embeddings, _Embeddings


class FastText(Embeddings):
    wiki_news_300 = _Embeddings(name=u'wiki_news_300',
                                dimensions=300,
                                corpus_size='16B',
                                vocabulary_size='1M',
                                download_url='https://dl.fbaipublicfiles.com/fasttext/vectors-english/'
                                             'wiki-news-300d-1M.vec.zip',
                                format='vec',
                                architecture='CBOW',
                                trained_data='Wikipedia 2017')

    common_crawl_300 = _Embeddings(name=u'common_crawl_300',
                                   dimensions=300,
                                   corpus_size='600B',
                                   vocabulary_size='2M',
                                   download_url='https://dl.fbaipublicfiles.com/fasttext/vectors-english/'
                                                'crawl-300d-2M.vec.zip',
                                   format='vec',
                                   architecture='CBOW',
                                   trained_data='Common Crawl (600B tokens)')

    _members = (
        wiki_news_300,
        common_crawl_300
    )

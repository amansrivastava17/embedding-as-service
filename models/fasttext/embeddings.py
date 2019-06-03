from models.utils import Embeddings, _Embeddings


class FastText(Embeddings):
    wiki_news_300 = _Embeddings(name=u'fastText-en-wiki-300',
                                dimensions=300,
                                corpus_size='16B',
                                vocabulary_size='1M',
                                download_url='',
                                format='vec',
                                architecture='CBOW',
                                trained_data='Wikipedia 2017')

    crawl_300 = _Embeddings(name=u'fastText-en-common-crawl-300',
                            dimensions=300,
                            corpus_size='600B',
                            vocabulary_size='2M',
                            download_url='',
                            format='vec',
                            architecture='CBOW',
                            trained_data='Common Crawl (600B tokens)')

    _members = (
        wiki_news_300,
        crawl_300
    )

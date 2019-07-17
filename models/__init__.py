# coding=utf-8
from __future__ import absolute_import
import collections

_Embeddings = collections.namedtuple('Embedding', ['name', 'dimensions', 'corpus_size', 'vocabulary_size',
                                                   'download_url', 'format', 'architecture', 'trained_data',
                                                   ])


class Embeddings(object):
    _members = ()
    _lookup_corpus_size = {m.name: m.corpus_size for m in _members}
    _lookup_vocab_size = {m.name: m.vocabulary_size for m in _members}
    _lookup_dimensions = {m.name: m.dimensions for m in _members}
    _lookup_format = {m.name: m.format for m in _members}
    _lookup_architecture = {m.name: m.architecture for m in _members}
    _lookup_trained_data = {m.name: m.trained_data for m in _members}
    _lookup_download_url = {m.name: m.download_url for m in _members}

    @classmethod
    def get_download_url(cls, name):
        return cls._lookup_download_url[name]

    @classmethod
    def get_corpus_size(cls, name):
        return cls._lookup_corpus_size[name]

    @classmethod
    def get_vocab_size(cls, name):
        return cls._lookup_vocab_size[name]

    @classmethod
    def get_dimensions(cls, name):
        return cls._lookup_dimensions[name]

    @classmethod
    def get_format(cls, name):
        return cls._lookup_format[name]

    @classmethod
    def get_architecture(cls, name):
        return cls._lookup_architecture[name]

    @classmethod
    def get_trained_data(cls, name):
        return cls._lookup_trained_data[name]

    @classmethod
    def get_all(cls):
        return [embedding._asdict() for embedding in cls._members]

    @classmethod
    def get_all_names(cls):
        return [embedding.name for embedding in cls._members]

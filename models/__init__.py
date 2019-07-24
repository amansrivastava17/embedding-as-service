# coding=utf-8
from __future__ import absolute_import
from typing import NamedTuple

MODELS_DIR = '.embeddings'


class Embedding(NamedTuple):
    name: str
    dimensions: int
    corpus_size: str
    vocabulary_size: str
    download_url: str
    format: str
    architecture: str
    trained_data: str
    language: str


# from gensim.models import KeyedVectors
# import _pickle as pickle
# import request
# import os
# import urllib
# import os.path
# from os import path
# import argparse
# import sys
# import time
# import numpy as np
# import zipfile
#
#
# def unzip(zip_dir, dest_dir):
#     zip_ref = zipfile.ZipFile(zip_dir, 'r')
#     zip_ref.extractall(dest_dir)
#     zip_ref.close()
#
#
# def preprocess(phrase):
#     import re
#     EMOJI_RANGES = {
#         u'regional_indicators': u'\U0001f1e6-\U0001f1ff',
#         u'misc_pictograms': u'\U0001f300-\U0001f5ff',
#         u'emoticons': u'\U0001f600-\U0001f64f',
#         u'transport': u'\U0001f680-\U0001f6ff',
#         u'supplemental': u'\U0001f900-\U0001f9ff\U0001f980-\U0001f984\U0001f9c0',
#         u'zero_width_separator': u'\U0000200d',
#         u'variation_selector': u'\U0000fe0f',
#         u'misc_dingbats': u'\U00002600-\U000027bf',
#         u'emoticon_skintones': u'\U0001f3fb-\U0001f3ff',
#         u'letterlike_symbols': u'\U00002100-\U0000214F',
#         u'arrows': u'\U00002190-\U000021FF',
#         u'miscellaneous_technical': u'\U00002300-\U000023FF',
#         u'enclosed_alphanumerics': u'\U00002460-\U000024FF',
#         u'geometric_shapes': u'\U000025A0-\U000025FF',
#         u'zws':u'\u200b'
#                 }
#     emoji_pattern = re.compile(u'[{}]+'.format(u''.join(EMOJI_RANGES.values())), flags=re.UNICODE)
#     phrase = re.sub(r'[\?,।|\.!\:\)\-…]+', ' ', phrase)
#     phrase = re.sub(" \d+", " ", phrase)
#     phrase = " ".join(phrase.split())
#     phrase = emoji_pattern.sub(u'', phrase)
#     stringlist = [x for x in phrase.split() if x.strip() != ""]
#     return stringlist
#
#
# # function to find the oov words
# def get_oovs(tokens):
#     return [token for token in tokens if token not in model]
#
#
# # function to get the sentence embeddings using the fasttext word embeddings
# def reporthook(count, block_size, total_size):
#     global start_time
#     if count == 0:
#         start_time = time.time()
#         return
#     duration = time.time() - start_time
#     progress_size = int(count * block_size)
#     speed = int(progress_size / (1024 * duration))
#     percent = int(count * block_size * 100 / total_size)
#     sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
#                     (percent, progress_size / (1024 * 1024), speed, duration))
#     sys.stdout.flush()
#
#
# def save(url, filename):
#     urllib.request.urlretrieve(url, filename, reporthook)
# # Given sentence or word return fixed length vector (using mean pooling method)
#
#
# class FasttextEncoder:
#     def __init__(self, language='en', download=False, path=None):
#         self.language = language
#         self.path = path
#         self.word_vectors = self._load_vector(language)
#
#     def _load_vector(self, language):
#         fullfilename = os.path.join(self.path + language, 'cc.{language}.300.vec.gz'.format(language=language))
#         if not path.exists(fullfilename):
#             os.makedirs(self.path + language)
#             #             testfile.retrieve("https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.%s.300.vec.gz"%language,
#             #                               fullfilename)
#             save("https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.%s.300.vec.gz" % language,
#                  fullfilename)
#         word_vectors = KeyedVectors.load_word2vec_format(fullfilename)
#
#         return word_vectors
#
#     def _encode_avg(self, tokens):
#         N = len(tokens)
#         in_vocab_tokens = [token for token in tokens if token in self.word_vectors]
#         if not in_vocab_tokens:
#             return np.zeros(shape=(300,), dtype=np.float32)
#         return np.sum(self.word_vectors[in_vocab_tokens], axis=0) / N
#
#     def get_sentence_vectors(self, sentence, pool='avg'):
#         if pool == 'avg':
#             return self._encode_avg(preprocess(sentence))

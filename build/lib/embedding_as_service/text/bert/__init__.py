from typing import List, Dict, Tuple, Optional, Union
import numpy as np

from embedding_as_service.text import Embedding
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm
from bert.tokenization import FullTokenizer

from embedding_as_service.utils import POOL_FUNC_MAP


class Embeddings(object):

    EMBEDDING_MODELS: List[Embedding] = [
                        Embedding(name=u'bert_base_uncased',
                                  dimensions=768,
                                  corpus_size='3300M',
                                  vocabulary_size='30522(sub-word)',
                                  download_url='https://storage.googleapis.com/tfhub-modules/'
                                               'google/bert_uncased_L-12_H-768_A-12/1.tar.gz',
                                  format='tar.gz',
                                  architecture='Transformer, Layers=12, Hidden = 768, heads = 12',
                                  trained_data='BooksCorpus(800M) English Wikipedia (2500M) words',
                                  language='en'),

                        Embedding(name=u'bert_base_cased',
                                  dimensions=768,
                                  corpus_size='3300M',
                                  vocabulary_size='30522(sub-word)',
                                  download_url='https://storage.googleapis.com/tfhub-modules/google/'
                                               'bert_cased_L-12_H-768_A-12/1.tar.gz',
                                  format='tar.gz',
                                  architecture='Transformer Layers=12, Hidden = 768, heads = 12',
                                  trained_data='BooksCorpus(800M) English Wikipedia (2500M) words',
                                  language='en'),

                        Embedding(name=u'bert_multi_cased',
                                  dimensions=768,
                                  corpus_size='3300M',
                                  vocabulary_size='30522 (sub-word)',
                                  download_url='https://storage.googleapis.com/tfhub-modules/google/'
                                               'bert_multi_cased_L-12_H-768_A-12/1.tar.gz',
                                  format='tar.gz',
                                  architecture='Transformer Layers=12, Hidden = 768, heads = 12',
                                  trained_data='BooksCorpus(800M) English Wikipedia (2500M) words',
                                  language='en'),

                        Embedding(name=u'bert_large_uncased',
                                  dimensions=1024,
                                  corpus_size='3300M',
                                  vocabulary_size='30522 (sub-word)',
                                  download_url='https://storage.googleapis.com/tfhub-modules/google/'
                                               'bert_uncased_L-24_H-1024_A-16/1.tar.gz',
                                  format='tar.gz',
                                  architecture='Transformer Layers=24, Hidden = 1024, heads = 16',
                                  trained_data='BooksCorpus(800M) English Wikipedia (2500M) words',
                                  language='en'),

                        Embedding(name=u'bert_large_cased',
                                  dimensions=1024,
                                  corpus_size='3300M',
                                  vocabulary_size='30522 (sub-word)',
                                  download_url='https://storage.googleapis.com/tfhub-modules/google/'
                                               'bert_cased_L-24_H-1024_A-16/1.tar.gz',
                                  format='tar.gz',
                                  architecture='Transformer Layers=24, Hidden = 1024, heads = 16',
                                  trained_data='BooksCorpus(800M) English Wikipedia (2500M) words',
                                  language='en')
                        ]

    EMBEDDING_MODELS: Dict[str, Embedding] = {embedding.name: embedding for embedding in EMBEDDING_MODELS}

    tokenizer: FullTokenizer = None

    def __init__(self):
        self.sess = tf.Session()
        self.bert_module = None
        self.model_name = None

    def create_tokenizer_from_hub_module(self, model_path: str):
        """Get the vocab file and casing info from the Hub module."""
        tokenization_info = self.bert_module(signature="tokenization_info", as_dict=True)
        vocab_file, do_lower_case = self.sess.run(
            [
                tokenization_info["vocab_file"],
                tokenization_info["do_lower_case"],
            ]
        )

        Embeddings.tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

    @classmethod
    def tokenize(cls, text):
        return cls.tokenizer.tokenize(text)

    @staticmethod
    def _model_single_input(text: Union[str, List[str]], max_seq_length: int, is_tokenized: bool = False
                            ) -> Tuple[List[int], List[int], List[int]]:
        tokens_a = text
        if not is_tokenized:
            tokens_a = Embeddings.tokenize(text)
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0: (max_seq_length - 2)]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = Embeddings.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        return input_ids, input_mask, segment_ids

    def load_model(self, model: str, model_path: str):
        self.bert_module = hub.Module(model_path)
        self.sess.run(tf.initializers.global_variables())
        self.create_tokenizer_from_hub_module(model_path)
        self.model_name = model
        print("Model loaded Successfully !")

    def encode(self, texts: Union[List[str], List[List[str]]],
               pooling: str,
               max_seq_length: int,
               is_tokenized: bool = False,
               **kwargs
               ) -> Optional[np.array]:
        input_ids, input_masks, segment_ids = [], [], []
        for text in tqdm(texts, desc="Converting texts to features"):
            input_id, input_mask, segment_id = self._model_single_input(text, max_seq_length, is_tokenized)
            input_ids.append(input_id)
            input_masks.append(input_mask)
            segment_ids.append(segment_id)

        bert_inputs = dict(
            input_ids=np.array(input_ids),
            input_mask=np.array(input_masks),
            segment_ids=np.array(segment_ids))

        bert_outputs = self.bert_module(bert_inputs, signature="tokens", as_dict=True)
        sequence_output = bert_outputs["sequence_output"]

        token_embeddings = self.sess.run(sequence_output)

        if not pooling:
            return token_embeddings
        else:
            if pooling not in POOL_FUNC_MAP.keys():
                print(f"Pooling method \"{pooling}\" not implemented")
                return None
            pooling_func = POOL_FUNC_MAP[pooling]
            pooled = pooling_func(token_embeddings, axis=1)
            return pooled

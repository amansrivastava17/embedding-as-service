from typing import List, Dict, Tuple, Optional, Union
import sentencepiece as spm
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import os

from embedding_as_service.text.xlnet.config import Flags
from embedding_as_service.text import Embedding

from embedding_as_service.text.xlnet.models.prepro_utils import preprocess_text, encode_ids, encode_pieces
from embedding_as_service.text.xlnet.models.data_utils import SEP_ID, CLS_ID
from embedding_as_service.text.xlnet.models import xlnet

from embedding_as_service.utils import POOL_FUNC_MAP

SEG_ID_A = 0
SEG_ID_B = 1
SEG_ID_CLS = 2
SEG_ID_SEP = 3
SEG_ID_PAD = 4


class Embeddings(object):

    EMBEDDING_MODELS: List[Embedding] = [
                        Embedding(name=u'xlnet_large_cased',
                                  dimensions=1024,
                                  corpus_size='32.89B',
                                  vocabulary_size='32000',
                                  download_url='https://storage.googleapis.com/xlnet/released_models/'
                                               'cased_L-24_H-1024_A-16.zip',
                                  format='zip',
                                  architecture='Transformer, 24-layer, 1024-hidden, 16-heads',
                                  trained_data='BooksCorpus(800M) English Wikipedia (2500M) words, Giga5 (16gb), '
                                               'ClueWeb 2012-B(19gb),  Common Crawl(78gb)',
                                  language='en'),

                        Embedding(name=u'xlnet_base_cased',
                                  dimensions=768,
                                  corpus_size='3.86B',
                                  vocabulary_size='32000',
                                  download_url='https://storage.googleapis.com/xlnet/released_models/'
                                               'cased_L-12_H-768_A-12.zip',
                                  format='zip',
                                  architecture='Transformer 12-layer, 768-hidden, 12-heads.',
                                  trained_data='BooksCorpus(800M) English Wikipedia (2500M) words',
                                  language='en')
                        ]

    EMBEDDING_MODELS: Dict[str, Embedding] = {embedding.name: embedding for embedding in EMBEDDING_MODELS}

    tokenizer: spm.SentencePieceProcessor = None
    mode_config_path: str = 'xlnet_config.json'
    sentence_piece_model_path: str = 'spiece.model'

    def __init__(self):
        self.xlnet_config = None
        self.run_config = None
        self.model_name = None
        self.sess = tf.Session()

    @staticmethod
    def load_tokenizer(model_path: str):
        """Get the vocab file and casing info from the Hub module."""
        sp_model = spm.SentencePieceProcessor()
        sp_model.Load(os.path.join(model_path, Embeddings.sentence_piece_model_path))
        Embeddings.tokenizer = sp_model

    @classmethod
    def tokenize(cls, text):
        text = preprocess_text(text, lower=False)
        return encode_pieces(cls.tokenizer, text)

    @staticmethod
    def _model_single_input(text: Union[str, List[str]], max_seq_length: int, is_tokenized: bool
                            ) -> Tuple[List[int], List[int], List[int]]:

        tokens_a = text
        if not is_tokenized:
            tokens_a = Embeddings.tokenize(text)

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0: (max_seq_length - 2)]

        tokens = []
        segment_ids = []

        tokens_a = [Embeddings.tokenizer.PieceToId(token) for token in tokens]
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(SEG_ID_A)
        tokens.append(SEP_ID)
        segment_ids.append(SEG_ID_A)

        tokens.append(CLS_ID)
        segment_ids.append(SEG_ID_CLS)

        input_ids = tokens

        # The mask has 0 for real tokens and 1 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [0] * len(input_ids)

        # Zero-pad up to the sequence length.
        if len(input_ids) < max_seq_length:
            delta_len = max_seq_length - len(input_ids)
            input_ids = [0] * delta_len + input_ids
            input_mask = [1] * delta_len + input_mask
            segment_ids = [SEG_ID_PAD] * delta_len + segment_ids

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        return input_ids, input_mask, segment_ids

    def load_model(self, model: str, model_path: str):
        model_path = os.path.join(model_path, next(os.walk(model_path))[1][0])
        self.xlnet_config = xlnet.XLNetConfig(json_path=os.path.join(model_path, Embeddings.mode_config_path))
        self.run_config = xlnet.create_run_config(is_training=True, is_finetune=True, FLAGS=Flags)
        self.load_tokenizer(model_path)
        self.model_name = model
        print("Model loaded Successfully !")

    def encode(self,
               texts: Union[List[str], List[List[str]]],
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

        # Construct an XLNet model
        xlnet_model = xlnet.XLNetModel(
            xlnet_config=self.xlnet_config,
            run_config=self.run_config,
            input_ids=np.array(input_ids, dtype=np.int32),
            seg_ids=np.array(segment_ids, dtype=np.int32),
            input_mask=np.array(input_masks, dtype=np.float32)
        )
        self.sess.run(tf.initializers.global_variables())

        # Get a sequence output
        sequence_output = xlnet_model.get_sequence_output()
        token_embeddings = self.sess.run(sequence_output)

        if not pooling:
            return token_embeddings
        else:
            if pooling not in POOL_FUNC_MAP.keys():
                raise NotImplementedError(f"Pooling method \"{pooling}\" not implemented")
            pooling_func = POOL_FUNC_MAP[pooling]
            pooled = pooling_func(token_embeddings, axis=1)
            return pooled

from typing import List, Dict, Tuple, Any, Optional
import sentencepiece as spm
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import os

from models.xlnet.config import Flags
from models import Embedding, TF_SESS

from xlnet.prepro_utils import preprocess_text, encode_ids
from xlnet.data_utils import SEP_ID, CLS_ID
from xlnet import xlnet


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
                                  vocabulary_size='na',
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
                                  vocabulary_size='na',
                                  download_url='https://storage.googleapis.com/xlnet/released_models/'
                                               'cased_L-12_H-768_A-12.zip',
                                  format='zip',
                                  architecture='Transformer 12-layer, 768-hidden, 12-heads.',
                                  trained_data='BooksCorpus(800M) English Wikipedia (2500M) words',
                                  language='en')
                        ]

    EMBEDDING_MODELS: Dict[str, Embedding] = {embedding.name: embedding for embedding in EMBEDDING_MODELS}

    tokenizer: spm.SentencePieceProcessor = None
    xlnet_config = None
    run_config = None
    model: str
    mode_config_path: str = 'xlnet_config.json'
    sentence_piece_model_path: str = 'spiece.model'

    @classmethod
    def load_tokenizer(cls, model_path: str):
        """Get the vocab file and casing info from the Hub module."""
        sp_model = spm.SentencePieceProcessor()
        sp_model.Load(os.path.join(model_path, cls.sentence_piece_model_path))
        cls.tokenizer = sp_model

    @classmethod
    def tokenize_fn(cls, text):
        text = preprocess_text(text, lower=False)
        return encode_ids(cls.tokenizer, text)

    @classmethod
    def _model_single_input(cls, text: str, max_seq_length: int) -> Tuple[List[int], List[int], List[int]]:
        tokens_a = cls.tokenize_fn(text)

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0: (max_seq_length - 2)]

        tokens = []
        segment_ids = []
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

    @classmethod
    def load_model(cls, model: str, model_path: str):
        model_path = os.path.join(model_path, next(os.walk(model_path))[1][0])
        cls.xlnet_config = xlnet.XLNetConfig(json_path=os.path.join(model_path, cls.mode_config_path))
        cls.run_config = xlnet.create_run_config(is_training=True, is_finetune=True, FLAGS=Flags)

        cls.load_tokenizer(model_path)
        cls.model = model
        print("Model loaded Successfully !")

    @classmethod
    def encode(cls, text: str, pooling: str = 'mean', **kwargs) -> Optional[np.array]:
        texts = [text]
        max_seq_length = kwargs.get('max_seq_length', 128)
        input_ids, input_masks, segment_ids = [], [], []
        for text in tqdm(texts, desc="Converting texts to features"):
            input_id, input_mask, segment_id = cls._model_single_input(text, max_seq_length)
            input_ids.append(input_id)
            input_masks.append(input_mask)
            segment_ids.append(segment_id)

        # Construct an XLNet model
        xlnet_model = xlnet.XLNetModel(
            xlnet_config=cls.xlnet_config,
            run_config=cls.run_config,
            input_ids=np.array(input_ids, dtype=np.int32),
            seg_ids=np.array(segment_ids, dtype=np.int32),
            input_mask=np.array(input_masks, dtype=np.float32))

        # Get a sequence output
        sequence_output = xlnet_model.get_sequence_output()

        if not pooling:
            return sequence_output

        if pooling == 'mean':
            return tf.reduce_mean(sequence_output, 0)

        elif pooling == 'max':
            return tf.reduce_max(sequence_output, 0)

        elif pooling == 'min':
            return tf.reduce_min(sequence_output, 0)

        elif pooling == 'mean_max':
            return tf.concat(values=[tf.reduce_mean(sequence_output, 0), tf.reduce_max(sequence_output, 0)], axis=0)
        else:
            print(f"Pooling method \"{pooling}\" not implemented")
            return None

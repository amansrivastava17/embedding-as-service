from keras.layers import (Input, Embedding, Dense, LSTM, TimeDistributed, Dropout, CuDNNLSTM)
from keras.models import Model
import os


from embedding_as_service.text.ulmfit.custom_layers.tied_embeddings import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def build_language_model(num_words=238462, embedding_size=400, rnn_sizes=(1150, 1150),
                         dropout=0.1, dropouth=0.3, dropouti=0.2, dropoute=0.1, wdrop=0.5,
                         tie_weights=True, use_qrnn=False, use_gpu=False, only_last=False):
    inp = Input(shape=(None,), name='input')
    emb = Embedding(num_words, embedding_size, name='embedding')
    emb_inp = emb(inp)
    emb_inp = Dropout(dropouti)(emb_inp)

    if use_qrnn:
        rnn = QRNN(rnn_sizes[0], return_sequences=True, window_size=2)(emb_inp)
        for rnn_size in rnn_sizes[1:]:
            rnn = QRNN(rnn_size, return_sequences=True, window_size=1)(rnn)
        if only_last:
            rnn = QRNN(embedding_size, return_sequences=False, window_size=1, name='final_rnn_layer')(rnn)
        else:
            rnn = QRNN(embedding_size, return_sequences=True, window_size=1, name='final_rnn_layer')(rnn)
    else:
        rnn_unit = CuDNNLSTM if use_gpu else LSTM
        rnn = rnn_unit(rnn_sizes[0], return_sequences=True, name='0_rnn_layer')(emb_inp)
        for i, rnn_size in enumerate(rnn_sizes[1:]):
            rnn = rnn_unit(rnn_size, return_sequences=True, name='{}_rnn_layer'.format(i + 1))(rnn)
        if only_last:
            rnn = rnn_unit(embedding_size, return_sequences=False, name='final_rnn_layer')(rnn)
        else:
            rnn = rnn_unit(embedding_size, return_sequences=True, name='final_rnn_layer')(rnn)

    if tie_weights:
        softmax_layer = TiedEmbeddingsTransposed(tied_to=emb, activation='softmax')
    else:
        softmax_layer = Dense(num_words, activation='softmax')

    if only_last:
        logits = softmax_layer(rnn)
    else:
        logits = TimeDistributed(softmax_layer)(rnn)

    out = Dropout(dropout)(logits)
    model = Model(inputs=inp, outputs=[out, rnn])
    return model

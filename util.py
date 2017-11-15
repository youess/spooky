# -*- coding: utf-8 -*-


import os
import numpy as np
import pandas as pd
from string import punctuation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from keras.layers import Input, Bidirectional, Dropout, Flatten, \
    Dense, LSTM, Conv1D, MaxPool1D, Embedding
from keras.models import Model


def read_csv(filename, loc='./data'):
    filename = os.path.join(loc, filename)
    return pd.read_csv(filename)


def clean_text(docs, nlp):

    texts = []
    for doc in docs:
        doc = nlp(doc)
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in ENGLISH_STOP_WORDS and tok not in punctuation]
        tokens = ' '.join(tokens)
        texts.append(tokens)
    text_vec = np.array([doc.vector for doc in nlp.pipe(texts, batch_size=500, n_threads=4)])
    return text_vec


def create_nn(arch='lstm_cnn', default_input_size=(300, 1)):

    if arch == 'lstm_cnn':

        max_features = 20000
        embedding_size = 128
        inputs = Input(shape=default_input_size)
        x = Embedding(max_features, embedding_size)(inputs)
        x = Dropout(.25)(x)
        x = Conv1D(filters=128, kernel_size=(2, 2), strides=2, padding='valid', activation='relu')(x)
        x = MaxPool1D(pool_size=2)(x)
        x = Bidirectional(LSTM(64, return_sequences=True), merge_mode='concat')(x)
        x = Dropout(.25)(x)
        x = Flatten()(x)
        outputs = Dense(3, activation='softmax')(x)

        m = Model(inputs=inputs, outputs=outputs, name="LSTM-CNN")
        return m

# -*- coding: utf-8 -*-

"""
Deal with text data and process it into word vector and char vector
"""

# import numpy as np
# import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# from sklearn.model_selection import train_test_split
# from util import read_data


# x_train, y_train, x_test = read_data(loc="../data")
# y_train = pd.get_dummies(y_train).values


# make word segmentation
def make_word_vector(texts, min_count=2, max_len=300):

    tokenizer = Tokenizer(lower=False, filters='')
    tokenizer.fit_on_texts(texts)
    filter_words_num = sum([1 for _, v in tokenizer.word_counts.items() if v >= min_count])

    tokenizer = Tokenizer(num_words=filter_words_num, lower=False, filters='')
    tokenizer.fit_on_texts(texts)
    docs = tokenizer.texts_to_sequences(texts)

    return pad_sequences(docs, maxlen=max_len)


# x_train = clean_text(x_train.values)

# make char segmentation
def make_char_vector(texts, max_len=600):
    pass

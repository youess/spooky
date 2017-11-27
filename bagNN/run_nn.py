# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import read_data
from textVectorPrecessoring import make_word_vector
from textCNN import TextCNN


data_x, data_y, test_x = read_data(loc='../data')
data_y = pd.get_dummies(data_y).values

docs = pd.concat([
    pd.concat([data_x, pd.Series([1 for _ in range(data_x.shape[0])])], axis=1),
    pd.concat([test_x, pd.Series([0 for _ in range(test_x.shape[0])])], axis=1)
])
docs.columns = ['text', 'tag']
is_train = (docs['tag'] == 1).values
print(docs.head())

docs = make_word_vector(docs['text'], min_count=2, max_len=300)

data_x = docs[is_train]
test_x = docs[~is_train]

# data_x = np.expand_dims(data_x, axis=2)

train_x, val_x, train_y, val_y = train_test_split(data_x, data_y, test_size=.15, random_state=333)

print(train_x.shape, val_x.shape, train_y.shape, val_y.shape)

nn = TextCNN(train_x, train_y, val_x, val_y, test_x)
nn.run()

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from util import read_csv
from collections import defaultdict
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
import keras
import keras.backend as K
from keras.layers import Dense, GlobalAveragePooling1D, Embedding
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

np.random.seed(7)

train = read_csv('train.csv')
x = train['text']
y = pd.get_dummies(train['author']).values

# check character distribution per author
counter = {name: defaultdict(int) for name in set(train.author)}
for (text, author) in zip(train.text, train.author):
    text = text.replace(' ', '')
    for c in text:
        counter[author][c] += 1

chars = set()
for v in counter.values():
    chars |= v.keys()

names = [author for author in counter.keys()]

print('c ', end='')
for n in names:
    print(n, end='   ')
print()
for c in chars:
    print(c, end=' ')
    for n in names:
        print(counter[n][c], end=' ')
    print()

"""
c EAP   MWS   HPL   
M 1065 415 645 
m 22792 20471 17622 
c 24127 17911 18338 
Z 23 2 51 
n 62636 50291 50879 
ü 1 0 5 
J 164 66 210 
î 1 0 0 
è 15 0 0 
E 435 445 281 
ê 28 0 2 
P 442 365 320 
j 683 682 424 
p 17422 12361 10965 
Å 0 0 1 
o 67145 53386 50996 
D 491 227 334 
" 2987 1469 513 
B 835 395 533 
α 0 0 2 
Ο 0 0 3 
A 1258 943 1167 
ç 1 0 0 
s 53841 45962 43915 
Σ 0 0 1 
Π 0 0 1 
é 47 0 15 
z 634 400 529 
' 1334 476 1710 
h 51580 43738 42770 
y 17001 14877 12534 
Y 282 234 111 
g 16088 12601 14951 
x 1951 1267 1061 
ï 0 0 7 
à 10 0 0 
X 17 4 5 
ë 0 0 12 
k 4277 3707 5204 
ö 16 0 3 
Ν 0 0 1 
K 86 35 176 
w 17507 16062 15554 
N 411 204 345 
G 313 246 318 
ἶ 0 0 2 
ä 1 0 6 
L 458 307 249 
r 51221 44042 40590 
l 35371 27819 30273 
U 166 46 94 
Q 21 7 10 
O 414 282 503 
δ 0 0 2 
u 26311 21025 19519 
f 22354 18351 16272 
. 8406 5761 5908 
; 1354 2662 1143 
H 864 669 741 
æ 36 0 10 
? 510 419 169 
T 2217 1230 1583 
v 9624 7948 6529 
Æ 1 0 4 
d 36862 35315 33366 
C 395 308 439 
Υ 0 0 1 
b 13245 9611 10636 
V 156 57 67 
, 17594 12045 8581 
ô 8 0 0 
R 258 385 237 
a 68525 55274 56815 
: 176 339 47 
F 383 232 269 
ñ 0 0 7 
i 60952 46080 44250 
q 1030 677 779 
W 739 681 732 
e 114885 97515 88259 
S 729 578 841 
I 4846 4917 3480 
t 82426 63142 62235 
â 6 0 0 
"""

# almost every char could be important.

# how to make features?

# single word count, also two continuous word could be important
docs = [word_tokenize(doc) for doc in train['text']]


def create_n_gram(tokens, n_max=2):
    n_grams = []
    for ni in range(2, n_max + 1):
        for wi in range(len(tokens) - ni + 1):
            e = '--'.join(tokens[wi:wi + ni])
            n_grams.append(e)
    return n_grams


"""
create_n_gram(word_tokenize('I want to get up early tommorror.'), 3)
Out[73]: 
['I--want',
 'want--to',
 'to--get',
 'get--up',
 'up--early',
 'early--tommorror',
 'tommorror--.',
 'I--want--to',
 'want--to--get',
 'to--get--up',
 'get--up--early',
 'up--early--tommorror',
 'early--tommorror--.']
"""

# should use stem or lemmatization, just skip it for a while

# join the documents with n gram
docs = [' '.join([' '.join(doc), ' '.join(create_n_gram(doc, 2))]) for doc in docs]


def create_docs(texts):
    docs = [word_tokenize(doc) for doc in texts]
    docs = [' '.join([' '.join(doc), ' '.join(create_n_gram(doc, 2))]) for doc in docs]
    return docs


min_count = 2

tokenizer = Tokenizer(lower=False, filters='')
tokenizer.fit_on_texts(docs)
num_words = sum([1 for _, v in tokenizer.word_counts.items() if v >= min_count])

tokenizer = Tokenizer(num_words=num_words, lower=False, filters='')
tokenizer.fit_on_texts(docs)
docs = tokenizer.texts_to_sequences(docs)

maxlen = 256
docs = pad_sequences(sequences=docs, maxlen=maxlen)

# fast-text classification
input_dim = np.max(docs) + 1
embedding_dims = 20


def create_model(embedding_dims=20, optimizer='adam'):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=embedding_dims))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


epochs = 25
x_train, x_test, y_train, y_test = train_test_split(docs, y, test_size=0.2)

model = create_model()
hist = model.fit(x_train, y_train,
                 batch_size=16,
                 validation_data=(x_test, y_test),
                 epochs=epochs,
                 verbose=1,
                 callbacks=[EarlyStopping(patience=5, monitor='val_loss')]
                 )

test = read_csv('test.csv')
test_docs = create_docs(test['text'])
tokenizer.fit_on_texts(test_docs)
test_docs = tokenizer.texts_to_sequences(test_docs)
test_docs = pad_sequences(test_docs, maxlen=maxlen)

y = model.predict_proba(test_docs)

results = read_csv('sample_submission.csv')
for i in range(y.shape[1]):
    results.iloc[:, i + 1] = np.clip(y[:, i])

# bad results
results.to_csv('data/sub/20171117-17-43-keras_kernel.csv', index=False)

# 执行错误，混在一起好像不行，word embeddings这个还不清楚

# merge all train and test vocab
train = read_csv('train.csv')
test = read_csv('test.csv')
test['author'] = None

train_y = pd.get_dummies(train['author']).values

data = pd.concat([train, test])

is_test = data['author'].isnull()

docs = create_docs(data['text'])
min_count = 2

tokenizer = Tokenizer(lower=False, filters='')
tokenizer.fit_on_texts(docs)
num_words = sum([1 for _, v in tokenizer.word_counts.items() if v >= min_count])

tokenizer = Tokenizer(num_words=num_words, lower=False, filters='')
tokenizer.fit_on_texts(docs)
docs = tokenizer.texts_to_sequences(docs)

maxlen = 256
docs = pad_sequences(sequences=docs, maxlen=maxlen)

docs = pd.DataFrame(docs)

train_x = docs.loc[(~is_test).values, :].values
test_x = docs.loc[is_test.values, :].values

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=.2, random_state=777)

# fast-text classification
input_dim = np.max(docs.values) + 1
embedding_dims = 20


def create_model(embedding_dims=20, optimizer='adam'):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=embedding_dims))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


model = create_model()
model.summary()
epochs = 25
hist = model.fit(train_x, train_y,
                 batch_size=16,
                 validation_data=(val_x, val_y),
                 epochs=epochs,
                 verbose=1,
                 callbacks=[EarlyStopping(patience=5, monitor='val_loss')]
                 )

y = model.predict_proba(test_x)

results = read_csv('sample_submission.csv')
for i in range(y.shape[1]):
    results.iloc[:, i + 1] = np.clip(y[:, i], 0.005, 0.995)

# good results
results.to_csv('data/sub/20171117-17-43-keras_kernel_v2.csv', index=False)

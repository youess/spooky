# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
import spacy
from string import punctuation
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords
from keras.layers import Input, Bidirectional, Dropout, Flatten, Dense, LSTM
from keras.models import Model
from keras.optimizers import Adam


def read_csv(filename, loc='./data'):
    filename = os.path.join(loc, filename)
    return pd.read_csv(filename)


def clean_text(docs):

    # nlp = spacy.load('en')
    texts = []
    for doc in docs:
        doc = nlp(doc)
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuation]
        tokens = ' '.join(tokens)
        texts.append(tokens)
    return pd.Series(texts)


train = read_csv('train.csv')

nlp = spacy.load('en')
data_x = clean_text(train['text'])

data_x_vec = np.array([doc.vector for doc in nlp.pipe(data_x, batch_size=500, n_threads=4)])

data_y = pd.get_dummies(train['author']).values

train_x, val_x, train_y, val_y = train_test_split(data_x_vec, data_y, test_size=.2)


def create_nn(arch='lstm'):

    if arch == 'lstm':

        inputs = Input(shape=(300, 1))
        x = Bidirectional(LSTM(64, return_sequences=True), merge_mode='concat')(inputs)

        x = Dropout(.2)(x)
        x = Flatten()(x)
        outputs = Dense(3, activation='softmax')(x)

        m = Model(inputs=inputs, outputs=outputs, name='LSTM')

    return m


model = create_nn()
model.summary()

if model.name == 'LSTM':
    train_x = np.expand_dims(train_x, axis=2)
    val_x = np.expand_dims(val_x, axis=2)

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['acc'])

epochs = 15
history = model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=epochs, batch_size=128, verbose=1)
# 可以用validation_split就不用sklearn中的划分了

test = read_csv('test.csv')

test_x = clean_text(test['text'])
test_x_vec = np.array([doc.vector for doc in nlp.pipe(test_x, batch_size=500, n_threads=4)])

test_y_pred = history.model.predict(np.expand_dims(test_x_vec, axis=2))

test_y_pred = pd.DataFrame(test_y_pred).clip(0.005, 0.995)
sub = pd.concat([test['id'], test_y_pred], axis=1)
sub.columns = ["id", "EAP", "HPL", "MWS"]
sub.to_csv('data/subs/submission_baseline.csv', index=False)

"""
data_x = data['text']
data_y = pd.get_dummies(data['author'])


pipe = Pipeline([
    # ('tf-idf', TfidfVectorizer()),
    ('vec count', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('nb', MultinomialNB())
    # ('svc', SVC(C=1.0))
])

pipe.fit(train_x, train_y)

val_y_pred = pipe.predict(val_x)

print('Validation dataset accuracy is: {:.3f}'.format(accuracy_score(val_y, val_y_pred)))
"""

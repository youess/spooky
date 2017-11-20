# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from util import read_data, write_sub
from sklearn.metrics import confusion_matrix, classification_report, log_loss
import spacy


train_x, train_y, test_x = read_data()
le = LabelEncoder()
train_y = le.fit_transform(train_y)
# train_y = pd.get_dummies(train_y)

clf = Pipeline([
    ('count', CountVectorizer(
        analyzer='word', binary=False, ngram_range=(1, 4),
        stop_words='english', min_df=2
    )),
    ('tf-idf', TfidfTransformer()),
    ('svc', SVC(C=1, probability=True))
])

# x_train, x_val, y_train, y_val = train_test_split(train_x, train_y, test_size=.2, random_state=777)

# clf.fit(x_train, y_train)


def print_eval(clf, x, y):

    yc = clf.predict(x)
    cm = confusion_matrix(y, yc)
    print('Confusion matrix is: \n{}'.format(cm))
    r = classification_report(y, yc)
    print('Classification report: \n{}'.format(r))
    yp = clf.predict_proba(x)
    print('logloss is: {}'.format(log_loss(y, yp)))


# print_eval(clf, x_val, y_val)

# use test corpus, cannot get high result. We need better approach
# 1. use train and test
# cv = CountVectorizer()
# corpus = train_x.tolist() + test_x.tolist()
# cnt = cv.fit_transform(corpus)                 # (27971, 153774), so many features

# 2. use wordvec to reduce the feature dimension
nlp = spacy.load('en')

corpus = pd.concat([
    pd.concat([train_x, pd.Series([1] * len(train_x))], axis=1),
    pd.concat([test_x, pd.Series([0] * len(test_x))], axis=1)
]).rename(columns={0: 'tag'})

is_train = corpus['tag'] == 1

corpus_cleaned = []
for doc in corpus['text']:
    d = [tok.lemma_.lower().strip() for tok in nlp(doc)]
    # tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
    n = len(d)
    tokens = []               # add two gram
    for i in range(1, n):
        tokens.append('--'.join(list(d[(i-1):(i+1)])))
    new_doc = ' '.join([doc, ' '.join(tokens)])
    corpus_cleaned.append(new_doc)

corpus_cleaned = [doc.vector for doc in nlp.pipe(corpus_cleaned)]
corpus_cleaned = np.array(corpus_cleaned)
train_x = corpus_cleaned[is_train.values, :]
test_x = corpus_cleaned[(~is_train).values, :]

x_train, x_val, y_train, y_val = train_test_split(train_x, train_y, test_size=.2, random_state=777)


for c in [0.1, .3, 1, 3, 10, 30]:
    clf = SVC(C=c, probability=True)
    print('C is: {}'.format(c))
    clf.fit(x_train, y_train)
    print_eval(clf, x_val, y_val)

# svm not performed good results.

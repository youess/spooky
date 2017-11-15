# -*- coding: utf-8 -*-


import os
# import gensim
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def read_csv(filename, loc='./data'):

    filename = os.path.join(loc, filename)
    return pd.read_csv(filename)


train = read_csv('train.csv')


class LemmaCountVectorizer(CountVectorizer):

    lemm = WordNetLemmatizer()

    def build_analyzer(self):
        analyzer = super(LemmaCountVectorizer, self).build_analyzer()
        return lambda doc: (self.lemm.lemmatize(w) for w in analyzer(doc))


# tf_vectorizer = LemmaCountVectorizer(
#     max_df=.95, min_df=2, stop_words='english', decode_error='ignore'
# )
# tf = tf_vectorizer.fit_transform(train_x)


def clean_text(text):

    # remove punctuation and tokenize text
    tokenizer = RegexpTokenizer('\w+')
    tokens = tokenizer.tokenize(text)

    # remove stopwords
    sw = stopwords.words('english')
    tokens = [w for w in tokens if w not in sw]

    # lemmatization and stemming
    lemm = WordNetLemmatizer()
    tokens = [lemm.lemmatize(w) for w in tokens]

    stemmer = PorterStemmer()
    tokens = [stemmer.stem(w) for w in tokens]
    return tokens


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(list(word2vec.values())[0])

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
        # return x


train_x = train['text'].apply(clean_text)
train_y = pd.get_dummies(train['author'])
model = Word2Vec(train_x, size=100)
w2v = dict(zip(model.wv.index2word, model.wv.syn0))
print('train word2vec done!')

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=.2)

etree_w2v = Pipeline([
    ('word2vec vectorize', MeanEmbeddingVectorizer(w2v)),
    ('extra trees', ExtraTreesClassifier(n_estimators=200))
])

etree_w2v.fit(train_x, train_y)

c = etree_w2v.predict(val_x)
print('Accuracy score: {:.3f}'.format(accuracy_score(val_y, c)))

test = read_csv('test.csv')

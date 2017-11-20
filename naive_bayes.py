# -*- coding: utf-8 -*-

import numpy as np
# import pandas as pd
from nltk import word_tokenize
# from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, log_loss
# from string import punctuation
from collections import Counter, OrderedDict
from itertools import chain
from util import read_csv, write_sub


def clean_texts(texts):

    docs = [word_tokenize(doc, language='english') for doc in texts]
    # remove stopwords
    sw = stopwords.words('english')

    # stem and lemmalization
    stemmer = PorterStemmer()
    lemmer = WordNetLemmatizer()
    new_docs = []
    for doc in docs:
        # for illustration
        words = [w for w in doc if w.lower() not in sw]
        words = [lemmer.lemmatize(w) for w in words]
        words = [stemmer.stem(w) for w in words]
        new_docs.append(words)
    return new_docs


train = read_csv('train.csv')
train_docs = clean_texts(train['text'])
test = read_csv('test.csv')
test_docs = clean_texts(test['text'])
# train_y = pd.get_dummies(train['author']).values
le = LabelEncoder()
train_y = le.fit_transform(train['author'])

# len(set(chain(*train_docs)))         # 18053

# counter the doc words
cnt = Counter()

# for word in chain(*train_docs):
docs = train_docs + test_docs
for word in chain(*docs):
    cnt[word.lower()] += 1


# pick most common words as feature name

# max_features_num = len(cnt)
# feature_words = cnt.most_common(max_features_num)
# feature_words = [x[0] for x in feature_words]

feature_words = [x[0] for x in cnt.items() if x[1] > 1]


# map train and test doc into features count
def feature_count_transform(docs, feature_words, freq=False):
    """
    Transform docs into count or term frequency
    :param docs: [['word', ...], ...]
    :param feature_words: ['fw1', 'fw2', ...]
    :return:
    """
    fn = len(feature_words)
    doc_cnt = []
    for doc in docs:
        feature_dict = OrderedDict(zip(feature_words, [0] * fn))
        for word in doc:
            if word.lower() in feature_words:
                feature_dict[word.lower()] += 1

        n = len(doc)
        if freq:
            for word in doc:
                if word.lower() in feature_words:
                    feature_dict[word.lower()] /= n
        doc_cnt.append(list(feature_dict.values()) + [n])
    return np.array(doc_cnt)


# train_doc_cnt = feature_count_transform(train_docs, feature_words, freq=True)
# test_doc_cnt = feature_count_transform(test_docs, feature_words, freq=True)
train_doc_cnt = feature_count_transform(train_docs, feature_words)
test_doc_cnt = feature_count_transform(test_docs, feature_words)

x_train, x_val, y_train, y_val = train_test_split(train_doc_cnt, train_y, test_size=.2, random_state=777)


clf = MultinomialNB(alpha=.6)
clf.fit(x_train, y_train)
y_val_pred = clf.predict_proba(x_val)
y_val_pred2 = clf.predict(x_val)


# -- f1, precision is about 0.62
print(confusion_matrix(y_val, y_val_pred2))
print(classification_report(y_val, y_val_pred2))
print(log_loss(y_val, y_val_pred))
y_test = clf.predict_proba(test_doc_cnt)

# v0. loss = 0.84, max_features_num=200
# v1. loss = 0.66, max_features_num=1000
# v2. loss = 0.58, max_features_num=2000
# v3. loss = 0.45, max_features_num=len(cnt)
# write_sub(y_test, 'nb_max_features_cnt_result.csv')

# v4. loss= 0.45 max_features_num=cnt > 1
# write_sub(y_test, 'nb_feature_cnt_more_1_res.csv')

# v5, loss = 1.01, max_features_num = cnt > 1, freq=False -> True
# write_sub(y_test, 'nb_feature_cnt_more_1_res_tf_idf.csv')

# v6, loss = 0.46, max_features_num = cnt > 1, freq=False, alpha=1.0 -> .4
# write_sub(y_test, 'nb_feature_cnt_more_1_res_alpha_0.4.csv')

# v7, loss = 0.5097, max_features_num = cnt > 1, freq=False, alpha=1.0 -> .1
# write_sub(y_test, 'nb_feature_cnt_more_1_res_alpha_0.1.csv')

# v8, loss = 0.4553, max_features_num = cnt > 1, freq=False, alpha=1.0 -> .6
write_sub(y_test, 'nb_feature_cnt_more_1_res_alpha_0.6.csv')

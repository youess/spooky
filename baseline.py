# -*- coding: utf-8 -*-

import os
import pandas as pd
import nltk
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


def read_csv(filename, loc='./data'):
    filename = os.path.join(loc, filename)
    return pd.read_csv(filename)


train = read_csv('train.csv')

'''
train.groupby('author').count()
Out[9]: 
          id  text
author            
EAP     7900  7900
HPL     5635  5635
MWS     6044  6044
'''

# 测试
text1 = train.text[0]

# 去除标点
punc_table = dict((ord(c), None) for c in punctuation)
text1 = text1.translate(punc_table)

# tokenization, 分词
token = nltk.word_tokenize(text1, language='english')
print(token)

# or both steps
# 测试
text1 = train.text[0]

tokenizer = nltk.tokenize.RegexpTokenizer('\w+')
token = tokenizer.tokenize(text1)

# stopword removal 去除停止词
stopwords = nltk.corpus.stopwords.words('english')
token_s = [w for w in token if w.lower() not in stopwords]
print(token_s)

# stemming(词干提取)
stemmer = nltk.stem.PorterStemmer()
print("The stem of following words ['running', 'runs', 'run'] is: {}".format(
    [stemmer.stem(e) for e in ['running', 'runs', 'run']]))
print("The stemmed form of leaves is: {}".format(stemmer.stem('leaves')))

# lemmatization(词形还原)
lemm = nltk.stem.WordNetLemmatizer()
lemm.lemmatize(stemmer.stem('leaves'))
lemm.lemmatize('leaves')


# 结合上述步骤
def token_extract(text):
    tokenizer = nltk.tokenize.RegexpTokenizer('\w+')
    stopwords = nltk.corpus.stopwords.words('english')
    lemm = nltk.stem.WordNetLemmatizer()
    stemmer = nltk.stem.PorterStemmer()

    tokens = tokenizer.tokenize(text)
    tokens = [w for w in tokens if w.lower() not in stopwords]
    tokens = [lemm.lemmatize(w) for w in tokens]
    tokens = [stemmer.stem(w) for w in tokens]
    return tokens


# train['text'] = train['text'].apply(token_extract)

# bag of words
sentences = ["I love to eat Burgers", "I love to eat Fries"]
vectorizer = CountVectorizer(min_df=0)
ss = vectorizer.fit_transform(sentences)
print(ss.shape)
print(vectorizer.get_feature_names())


# 预处理有一个更好的方式进行处理, 利用sklearn当中实现的函数，但是没有lemmatizer
class LemmaCountVectorizer(CountVectorizer):
    lemm = nltk.stem.WordNetLemmatizer()
    stem = nltk.stem.PorterStemmer()

    def build_analyzer(self):
        analyzer = super(LemmaCountVectorizer, self).build_analyzer()
        return lambda doc: (self.stem.stem(self.lemm.lemmatize(w)) for w in analyzer(doc))


tf_vectorizer = LemmaCountVectorizer(
    max_df=.95, min_df=2,
    stop_words='english',
    decode_error='ignore'
)
tf = tf_vectorizer.fit_transform(train['text'].tolist())
tf_feature_names = tf_vectorizer.get_feature_names()
# using LDA, latent dirichlet allocation un-supervised learning algorithm

lda = LatentDirichletAllocation(
    n_topics=6,
    max_iter=5,
    learning_method='online',
    learning_offset=50.,
    random_state=0
)
lda.fit(tf)


# print top words
def print_top_words(model, feature_names, n_top_words):

    for index, topic in enumerate(model.components_):
        msg = "\nTopic #{}:".format(index)
        msg += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]])
        print(msg)
        print('='*70)


n_top_words = 10
print_top_words(lda, tf_feature_names, n_top_words)

first_topic = lda.components_[0]

first_topic_words = [tf_feature_names[i] for i in first_topic.argsort()[:-50 - 1:-1]]

# Generating the word-cloud with the values under the category dataframe
first_cloud = WordCloud(
    stopwords=STOPWORDS,
    background_color='black',
    width=2500,
    height=1800
).generate(" ".join(first_topic_words))
plt.imshow(first_cloud)
plt.axis('off')
plt.show()

# 20 topics
topic_words = []
topic_top_word_num = 50
for index, t in enumerate(lda.components_):
    ti = t.argsort()
    top_ti = ti[:-topic_top_word_num:-1]
    topic_w = [(tf_feature_names[i], t[ti[i]], 'topic_{}'.format(index)) for i in top_ti]
    topic_w = sorted(topic_w, key=lambda x: x[1])
    topic_words.append(pd.DataFrame(topic_w, columns=['word', 'score', 'topic']))

topic_words = pd.concat(topic_words)

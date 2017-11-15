# -*- coding: utf-8 -*-

import spacy
import numpy as np
import pandas as pd
from util import read_csv, clean_text, create_nn
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime


train = read_csv('train.csv')
test = read_csv('test.csv')

nlp = spacy.load('en')
train_x = clean_text(train['text'], nlp)
test_x = clean_text(test['text'], nlp)

train_y = pd.get_dummies(train['author']).values
test_y = pd.get_dummies(test['author']).values

lstm = create_nn()

lstm.summary()

if lstm.name == 'lstm-cnn':
    train_x = np.expand_dims(train_x, axis=2)
    test_x = np.expand_dims(test_x, axis=2)

lstm.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['acc'])

epochs = 80
batch_size = 128

timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
model_file = 'data/checkpoints/{}_lstm_cnn_checkpoint.hdf5'.format(timestamp)
es = EarlyStopping(monitor='val_loss', patience=10)
cp = ModelCheckpoint(model_file, monitor='val_loss', verbose=0, save_best_only=True)

history = lstm.fit(train_x, train_y, validation_split=.2,
                   epochs=epochs, batch_size=batch_size,
                   callbacks=[es, cp], verbose=1)

eval = [pd.Series(history.history['val_loss']),
        pd.Series(history.history['val_acc'])]
eval = pd.concat(eval, axis=1)
eval.to_csv('data/model_eval.csv', index=False)

test_y_pred = history.model.predict(test_x)

test_y_pred = pd.DataFrame(test_y_pred).clip(0.005, 0.995)
sub = pd.concat([test['id'], test_y_pred], axis=1)
sub.columns = ["id", "EAP", "HPL", "MWS"]
sub.to_csv('data/subs/submission_baseline_{}.csv'.format(timestamp), index=False)

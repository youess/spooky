# -*- coding: utf-8 -*-

"""

    x0        x1       x2
    |         |         |
    |         |         |
--> LSTM --> LSTM --> LSTM -->                       |
    |         |         |                            |
    |         |         |                            |
<-- LSTM <-- LSTM <--  LSTM                          |       bi-directional layer
    |                                                |
    |                                                |
    concat   concat    concat                        |
    |
    |
    ---------------------------------------
     K-MaxPooling
     |
     softmax

"""

import numpy as np
from keras.layers import LSTM, Bidirectional, Dense, Dropout, Input, MaxPool1D, Flatten
from keras.models import Model
from keras.optimizers import Adam
from base import BaseModel


class TextRNN(BaseModel):

    def __init__(self, train_x, train_y, val_x, val_y, test_x, **kwargs):
        super().__init__(train_x, train_y, val_x, val_y, test_x, **kwargs)
        self.x_train = np.expand_dims(train_x, axis=2)
        self.x_val = np.expand_dims(val_x, axis=2)
        self.x_test = np.expand_dims(test_x, axis=2)

    def create_model(self):
        print(self.x_train.shape)
        inputs = Input(shape=(self.x_train.shape[1], 1, ))
        x = Bidirectional(LSTM(32, return_sequences=True), merge_mode='concat')(inputs)
        x = Dropout(0)(x)
        x = MaxPool1D()(x)
        x = Flatten()(x)
        outputs = Dense(3, activation='softmax')(x)
        m = Model(inputs=inputs, outputs=outputs, name='RNN-LSTM')
        m.compile(optimizer=Adam(self.lr), loss='categorical_crossentropy', metrics=['accuracy'])
        m.summary()
        return m

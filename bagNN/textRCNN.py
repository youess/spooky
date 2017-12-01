# -*- coding: utf-8 -*-

"""

embedding -> bi-LSTM -> bi-LSTM -> concat --> K-MaxPooling -> Conv(2) -> SoftMax
    |                                |
    ----------------------------------
"""

import numpy as np
from utils import KMaxPooling
from base import BaseModel
from keras.layers import Embedding, LSTM, Bidirectional, Conv1D, Input, Dropout, Add, Dense, Flatten, MaxPool1D
from keras.models import Model
from keras.optimizers import Adam


class TextRCNN(BaseModel):

    def __init__(self, train_x, train_y, val_x, val_y, test_x, **kwargs):
        super().__init__(train_x, train_y, val_x, val_y, test_x, **kwargs)
        # self.x_train = np.expand_dims(train_x, axis=2)
        # self.x_val = np.expand_dims(val_x, axis=2)
        # self.x_test = np.expand_dims(test_x, axis=2)

    def create_model(self):
        
        inputs = Input(shape=(self.x_train.shape[1], ))
        # print('Input shape: {}'.format(inputs._keras_shape))
        em = Embedding(input_dim=self.input_max_size, output_dim=100)(inputs)
        # print('Embedding shape: {}'.format(em._keras_shape))
        x = Dropout(0.1)(em)
        x1 = Bidirectional(LSTM(32, return_sequences=True), merge_mode='concat')(x)
        x2 = Bidirectional(LSTM(32, return_sequences=True), merge_mode='concat')(x)
        x = Add()([x1, x2])
        # import keras.backend as K
        # print(str(K.ndim(x)))

        # print('Concat 2 Bidirectional LSTM shape: {}'.format(x._keras_shape))
        # x = KMaxPooling(k=3)(x)
        x = MaxPool1D()(x)
        # print('K max pooling shape: {}'.format(x._keras_shape))
        # print(str(K.ndim(x)))
        x = Conv1D(filters=40, kernel_size=2, padding='same')(x)
        x = Flatten()(x)
        x = Dropout(.5)(x)
        # x = Dense(100, activation='relu')(x)
        outputs = Dense(3, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(self.lr), loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

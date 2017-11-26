# -*- coding: utf-8 -*-

"""
Text -> embedding -> NN classifier

                            conv(1) -> BN -> RELU -> conv(1) -> BN -> RELU -> MaxPooling
                            conv(2) -> BN -> RELU -> conv(2) -> BN -> RELU -> MaxPooling
Text Vector -> Embedding -> conv(3) -> BN -> RELU -> conv(3) -> BN -> RELU -> MaxPooling  -> concat -> softmax
                            conv(4) -> BN -> RELU -> conv(4) -> BN -> RELU -> MaxPooling
                            conv(5) -> BN -> RELU -> conv(5) -> BN -> RELU -> MaxPooling
"""

import numpy as np
from keras.layers import Conv1D, BatchNormalization, Activation, MaxPool1D, Add, Embedding, Dense
from keras.models import Model, Input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from .utils import add_timestamp, write_sub


class TextCNN(object):

    def __init__(self, train_x, train_y, val_x, val_y, test_x, **kwargs):
        """
        Initialization of text cnn model
        :param data_x: numpy array as text vector
        :param data_y: numpy array as classification number
        """
        self.x = train_x
        self.y = train_y
        self.x_val = val_x
        self.y_val = val_y
        self.x_test = test_x
        self.embed_size = kwargs.get('embed_size', 20)
        self.batch_size = kwargs.get('batch_size', 32)
        self.epochs_num = kwargs.get('epochs', 5)
        self.input_max_size = np.max(train_x) + 1
        self.lr = kwargs.get('lr', 0.001)
        self.patience_num = kwargs.get('patience_num', 5)
        self.model_file = add_timestamp('_text_cnn.h5')
        self.sub_file = self.model_file.strip('.h5') + '_text_cnn.csv'
        self.model = None

    @classmethod
    def conv1_max_pooling_layer(cls, embed_x, filters=40, kernel_size=1, padding='same'):

        x = Conv1D(filters=filters, kernel_size=kernel_size, padding=padding)(embed_x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool1D()(x)
        return x

    def train(self, input_shape):

        input = Input(shape=input_shape)
        ex = Embedding(input_dim=self.input_max_size, output_dim=100)(input)
        x1 = self.conv1_max_pooling_layer(ex, filters=40, kernel_size=1)
        x2 = self.conv1_max_pooling_layer(ex, filters=40, kernel_size=2)
        x3 = self.conv1_max_pooling_layer(ex, filters=40, kernel_size=3)
        x4 = self.conv1_max_pooling_layer(ex, filters=40, kernel_size=4)
        x5 = self.conv1_max_pooling_layer(ex, filters=40, kernel_size=5)

        x = Add()[x1, x2, x3, x4, x5]
        output = Dense(100, activation='softmax')
        self.model = Model(inputs=input, outputs=output, name='text-cnn')
        self.model.compile(optimizer=Adam(self.lr), loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()

        # callbacks
        es = EarlyStopping(monitor='val_loss', patience=self.patience_num)
        cp = ModelCheckpoint(self.model_file, monitor='val_loss', save_best_only=True)
        self.model.fit(
            self.x, self.y,
            batch_size=self.batch_size,
            validation_data=(self.x_val, self.y_val),
            epochs=self.epochs_num,
            verbose=1,
            callbacks=[es, cp]
        )

    def predict(self, x):

        return self.model.predict_proba(x)

    def write_sub(self):

        if self.model is not None:
            y_test = self.predict(self.x_test)
            write_sub(y_test, filename=self.sub_file, loc='../data/sub')

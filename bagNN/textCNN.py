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
from keras.layers import Conv1D, BatchNormalization, Activation, \
    MaxPool1D, Add, Embedding, Dense, Flatten, Dropout, AveragePooling1D
from keras.models import Model, Input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers
from utils import add_timestamp, write_sub


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
        self.embed_size = kwargs.get('embed_size', 30)
        self.batch_size = kwargs.get('batch_size', 64)
        self.epochs_num = kwargs.get('epochs', 80)
        self.input_max_size = max(np.max(train_x), np.max(val_x), np.max(test_x)) + 1
        self.lr = kwargs.get('lr', 0.001)
        self.patience_num = kwargs.get('patience_num', 5)
        self.model_file = add_timestamp('_text_cnn.h5')
        self.sub_file = self.model_file.strip('.h5') + '_text_cnn.csv'
        self.model = None

    @classmethod
    def conv1_max_pooling_layer(cls, embed_x, filters=40, kernel_size=1, l2_alpha=0, padding='same'):

        x = Conv1D(filters=filters, kernel_size=kernel_size, padding=padding,
                   kernel_regularizer=regularizers.l2(l2_alpha))(embed_x)
        # x = Dropout(.3)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        # x = MaxPool1D()(x)
        x = AveragePooling1D()(x)
        return x

    def train(self):

        inputs = Input(shape=(self.x.shape[1], ))
        ex = Embedding(input_dim=self.input_max_size, output_dim=100)(inputs)
        ex = Dropout(.2)(ex)
        x1 = self.conv1_max_pooling_layer(ex, filters=50, kernel_size=1)
        x2 = self.conv1_max_pooling_layer(ex, filters=50, kernel_size=2)
        x3 = self.conv1_max_pooling_layer(ex, filters=50, kernel_size=3)
        x4 = self.conv1_max_pooling_layer(ex, filters=50, kernel_size=4)
        x5 = self.conv1_max_pooling_layer(ex, filters=50, kernel_size=5)

        x = Add()([x1, x2, x3, x4, x5])
        x = Flatten()(x)
        x = Dropout(.5)(x)
        # x = Dense(100, activation='relu')(x)
        outputs = Dense(3, activation='softmax')(x)

        self.model = Model(inputs=inputs, outputs=outputs, name='text-cnn')
        self.model.compile(optimizer=Adam(self.lr), loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()

        # callbacks
        es = EarlyStopping(monitor='val_loss', patience=self.patience_num)
        cp = ModelCheckpoint('model/' + self.model_file, monitor='val_loss', save_best_only=True)
        self.model.fit(
            self.x, self.y,
            batch_size=self.batch_size,
            validation_data=(self.x_val, self.y_val),
            epochs=self.epochs_num,
            verbose=1,
            callbacks=[es, cp]
        )

    def predict(self, x):

        return self.model.predict(x)

    def write_sub(self):

        if self.model is not None:
            y_test = self.predict(self.x_test)
            write_sub(y_test, filename=self.sub_file)

    def run(self):
        # loss: 0.1103 - acc: 0.9627 - val_loss: 1.4722 - val_acc: 0.7215
        self.train()
        self.write_sub()

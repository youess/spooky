# -*- coding: utf-8 -*-

"""
                conv(1)  -------------------------->
embedding -->   conv(1) -> BN -> RELU -> conv(3) -->  concat --> BN -> RELU -> softmax
                conv(3) -> BN -> RELU -> conv(5) -->
                conv(3)  -------------------------->
"""


from base import BaseModel
from keras.layers import Embedding, BatchNormalization, Conv1D, Input, Dropout, \
    Add, Dense, Flatten, Activation
from keras.models import Model
from keras.optimizers import Adam


class TextInception(BaseModel):
    def __init__(self, train_x, train_y, val_x, val_y, test_x, **kwargs):
        super().__init__(train_x, train_y, val_x, val_y, test_x, **kwargs)

    def create_model(self):
        inputs = Input(shape=(self.x_train.shape[1], ))
        # print('Input shape: {}'.format(inputs._keras_shape))
        em = Embedding(input_dim=self.input_max_size, output_dim=100)(inputs)
        x = Dropout(0.2)(em)
        x1 = Conv1D(40, 1, padding='same')(x)
        x2 = BatchNormalization()(x1)
        x2 = Activation('relu')(x2)
        x2 = Conv1D(40, 3, padding='same')(x2)
        x3 = Conv1D(40, 3, padding='same')(x)
        x4 = BatchNormalization()(x3)
        x4 = Activation('relu')(x4)
        x4 = Conv1D(40, 3, padding='same')(x4)

        x1 = Dropout(.7)(x1)
        x2 = Dropout(.7)(x2)
        x3 = Dropout(.7)(x3)
        x4 = Dropout(.7)(x4)
        x = Add()([x1, x2, x3, x4])
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Flatten()(x)
        x = Dropout(.3)(x)

        outputs = Dense(3, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs, name='text inception')
        model.compile(optimizer=Adam(self.lr), loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

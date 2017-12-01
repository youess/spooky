# -*- coding: utf-8 -*-

from base import BaseModel
from keras.layers import GlobalAveragePooling1D, Input, Embedding, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam


class FastText(BaseModel):
    def __init__(self, train_x, train_y, val_x, val_y, test_x, **kwargs):
        super().__init__(train_x, train_y, val_x, val_y, test_x, **kwargs)

    def create_model(self):
        inputs = Input(shape=(self.x_train.shape[1],))
        em = Embedding(input_dim=self.input_max_size, output_dim=20)(inputs)
        x = GlobalAveragePooling1D()(em)
        x = Dropout(self.dropout_ratio)(x)
        outputs = Dense(3, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs, name='fastText')
        model.compile(optimizer=Adam(self.lr), loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

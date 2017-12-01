# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from utils import write_sub, add_timestamp
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model


class BaseModel(object):

    def __init__(self, train_x, train_y, val_x, val_y, test_x, **kwargs):
        self.x_train = train_x
        self.y_train = train_y
        self.x_val = val_x
        self.y_val = val_y
        self.x_test = test_x
        self.model_file = add_timestamp('{}.h5'.format(self.__class__.__name__))
        self.sub_file = self.model_file.strip('.h5') + '.csv'
        self.plot_file = 'model/' + self.model_file.strip('.h5') + '.png'
        self.model_file = 'model/' + self.model_file
        self.model = None

        self.input_max_size = max(np.max(train_x), np.max(val_x), np.max(test_x)) + 1
        self.epochs_num = kwargs.get('epochs_num', 15)
        self.batch_size = kwargs.get('batch_size', 16)
        self.lr = kwargs.get('learning_rate', 0.001)
        self.patience_num = kwargs.get('patience_num', 5)
        self.dropout_ratio = 0

    def create_model(self):
        raise NotImplemented

    def train(self):

        self.model = self.create_model()

        # callbacks
        es = EarlyStopping(monitor='val_loss', patience=self.patience_num)
        cp = ModelCheckpoint(self.model_file, monitor='val_loss', save_best_only=True)
        return self.model.fit(
            self.x_train, self.y_train,
            batch_size=self.batch_size,
            validation_data=(self.x_val, self.y_val),
            epochs=self.epochs_num,
            verbose=1,
            callbacks=[es, cp]
        )

    def predict(self, x):
        if self.model_file is not None:
            model = load_model(self.model_file)
            return model.predict(x)
        raise NotImplemented

    def write_sub(self):

        y_test = self.predict(self.x_test)
        write_sub(y_test, filename=self.sub_file)

    @classmethod
    def plot_train_curve(cls, history, filename=None):
        h = history.history

        figure = plt.gcf()

        plt.subplot(211)
        plt.plot(h['acc'])
        plt.plot(h['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        plt.subplot(212)
        plt.plot(h['loss'])
        plt.plot(h['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        # plt.show()
        # plt.draw()

        figure.savefig(filename, dpi=200, format='png')

    def run(self):
        history = self.train()
        self.write_sub()
        self.plot_train_curve(history, self.plot_file)

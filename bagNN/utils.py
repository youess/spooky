# -*- coding: utf-8 -*-

import os
import pandas as pd
from datetime import datetime
# Implement k max pooling from source git:https://github.com/fchollet/keras/issues/373
from keras.engine import Layer, InputSpec
from keras.layers import Flatten
import tensorflow as tf
import keras.backend as K


def add_timestamp(text):
    tsp = datetime.now()
    return tsp.strftime('%Y-%m-%d_%H') + "_" + text


def read_csv(filename, loc='./data'):
    filename = os.path.join(loc, filename)
    return pd.read_csv(filename)


def read_data(loc='./data'):
    train = read_csv('train.csv', loc)
    test = read_csv('test.csv', loc)
    return train['text'], train['author'], test['text']


def write_sub(y_predict, filename, loc='./sub'):
    filename = os.path.join(loc, filename)
    sub = read_csv('sample_submission.csv', loc=loc)
    sub.iloc[:, 1:] = y_predict
    sub.to_csv(filename, index=False, float_format='%.4f')


class KMaxPooling(Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """

    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.k * input_shape[1], input_shape[2]

    def call(self, inputs, **kwargs):

        print(inputs.get_shape())
        inputs = K.expand_dims(inputs, 2)
        print('After expand dim input shape: {}'.format(inputs.get_shape()))
        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_input = tf.transpose(inputs, (0, 3, 2, 1))
        print('Shift input shape: {}'.format(shifted_input.get_shape()))
        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(shifted_input, k=self.k, sorted=True, name=None)[0]
        print(top_k.get_shape())

        top_k = tf.transpose(top_k, (0, 2, 3, 1))
        # return flattened output
        # return Flatten()(top_k)
        # top_k = Flatten()(top_k)
        print(top_k.get_shape())

        '''
        Entry Inputs: (?, ?, 128)
        After Inputs: (?, ?, 1, 128)
        Output shape: (?, ?, 1, 128)
        Output squeeze shape: (?, ?, 128)
        K max pooling shape: (None, 150, 128)
        '''
        return K.squeeze(top_k, axis=1)
        # return K.expand_dims(Flatten()(top_k), 2)

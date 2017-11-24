#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf


class FastText(object):
    """
    A similar fast text arch for text classification.
    """

    def __init__(self, **kwargs):

        # length of input sentence, it should be padded to the same length
        self.seq_len = kwargs.get('seq_len', 300)

        # output layer size
        self.num_classes = kwargs.get('num_classes', 3)

        # vocabulary size
        self.vocab_size = kwargs.get('vocab_size', None)

        # embedding size
        self.embed_size = kwargs.get('embed_size', 30)

        # Add placeholder
        self.input_x = tf.placeholder(tf.int32, [None, self.seq_len], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # keep tracking of l2 regularization loss
        self.l2_loss = tf.constant(0.0)

        self.embedding_layer()

    def embedding_layer(self):

      with tf.device('/cpu:0'), tf.name_scope("embedding"):
        self.W = tf.Variable(tf.random_uniform([self.vocab_size]), name="W")
        self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
        self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

    def global_avg_pooling(self):
      pass

#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   cnn.py
#        \author   chenghuige  
#          \date   2016-02-24 16:02:40.758523
#   \Description  
# ==============================================================================
import tensorflow as tf
import melt

from tensorflow.contrib import skflow

#--------------------------------------------- simple cnn model, now the most simple one, padding to make each post with the same length
class CharCnnOptions(object):
    def __init__(self):
        self.emb_dim = 128


class CharCnn(object):
    def __init__(self, options = CharCnnOptions()):
        self.options = options

    def forward(self, trainer):
        options = self.options
        sequence_length = trainer.num_features
        emb_dim = options.emb_dim


        X = trainer.X
        MAX_DOCUMENT_LENGTH = 220

        N_FILTERS = 128
        FILTER_SHAPE1 = [10, emb_dim]
        FILTER_SHAPE2 = [10, N_FILTERS]
        POOLING_WINDOW = 4
        POOLING_STRIDE = 2

        """Character level convolutional neural network model to predict classes."""
        byte_list = tf.reshape(skflow.ops.one_hot_matrix(X,  emb_dim), 
            [-1, MAX_DOCUMENT_LENGTH,  emb_dim, 1])
        with tf.variable_scope('CNN_Layer1'):
            # Apply Convolution filtering on input sequence.
            conv1 = skflow.ops.conv2d(byte_list, N_FILTERS, FILTER_SHAPE1, padding='VALID')
            # Add a RELU for non linearity.
            conv1 = tf.nn.relu(conv1)
            # Max pooling across output of Convlution+Relu.
            pool1 = tf.nn.max_pool(conv1, ksize=[1, POOLING_WINDOW, 1, 1], 
                strides=[1, POOLING_STRIDE, 1, 1], padding='SAME')
            # Transpose matrix so that n_filters from convolution becomes width.
            pool1 = tf.transpose(pool1, [0, 1, 3, 2])
        with tf.variable_scope('CNN_Layer2'):
            # Second level of convolution filtering.
            conv2 = skflow.ops.conv2d(pool1, N_FILTERS, FILTER_SHAPE2,
                padding='VALID')
            # Max across each filter to get useful features for classification.
            pool2 = tf.squeeze(tf.reduce_max(conv2, 1), squeeze_dims=[1])
        # Apply regular WX + B and classification.

        # Final (unnormalized) scores and predictions
        W = tf.Variable(tf.truncated_normal([N_FILTERS, 1], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[1]), name="b")
        return tf.matmul(pool2, W) + b

 

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
class RnnOptions(object):
    def __init__(self):
        self.emb_dim = 256

        self.dropout_keep_prob = 1.0


class Rnn(object):
    def __init__(self, options = RnnOptions()):
        self.options = options

    def forward(self, trainer):
        options = self.options
        sequence_length = trainer.num_features
        emb_dim = options.emb_dim

        # Convert indexes of words into embeddings.
        # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
        # maps word indexes of the sequence into [batch_size, sequence_length,
        # EMBEDDING_SIZE].
        n_words = trainer.total_features
        EMBEDDING_SIZE = emb_dim
        MAX_DOCUMENT_LENGTH = 220
        X = trainer.X
        word_vectors = skflow.ops.categorical_variable(X, n_classes=n_words,
            embedding_size=EMBEDDING_SIZE, name='words')
        # Split into list of embedding per word, while removing doc length dim.
        # word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
        word_list = skflow.ops.split_squeeze(1, MAX_DOCUMENT_LENGTH, word_vectors)
        # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
        cell = tf.nn.rnn_cell.GRUCell(EMBEDDING_SIZE)
        # Create an unrolled Recurrent Neural Networks to length of
        # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.
        _, encoding = tf.nn.rnn(cell, word_list, dtype=tf.float32)
        # Given encoding of RNN, take encoding of last step (e.g hidden size of the
        # neural network of last step) and pass it as features for logistic
        # regression over output classes.

        # Final (unnormalized) scores and predictions
        W = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[1]), name="b")
        return tf.matmul(encoding, W) + b

 

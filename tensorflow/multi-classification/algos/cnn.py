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
from melt import NUM_CLASSES
#--------------------------------------------- simple cnn model, now the most simple one, padding to make each post with the same length
class CnnOptions(object):
    def __init__(self):
        #self.emb_dim = 128
        #self.emb_dim = 32

        #self.emb_dim = 300
        
        #self.emb_dim = 128
        #self.emb_dim = 512
        #self.emb_dim = 1500

        #self.emb_dim = 800

        self.emb_dim = 512

        #self.filter_sizes = [3, 4, 5]
        #self.filter_sizes = [5, 6, 7]
        #self.filter_sizes = [6, 7, 8]
        #self.filter_sizes = [5, 15, 27]
        #self.filter_sizes = [5, 15, 27, 51]
        #self.filter_sizes = [3, 5, 9]
        #self.filter_sizes = [3, 9, 15]
        #self.filter_sizes = [3, 7, 15]
        #self.filter_sizes = [3, 7, 5, 11, 15]
        
        #self.filter_sizes = [11, 15, 17]

        self.filter_sizes = [3, 5, 7]

        self.num_filters = 128
        #self.num_filters = 64
        #self.num_filters = 200
        #self.num_filters = 256
        #self.num_filters = 300
        #self.num_filters = 32
        
        self.dropout_keep_prob = 0.2
        #self.dropout_keep_prob = 0.8

        #self.dropout_keep_prob = 1.0

#@TODO @FIXME Algos save not using cpickle or find ways of cpickle dump without some attributes?
#http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
class Cnn(object):
    def __init__(self, options = CnnOptions()):
        self.options = options

    def forward(self, trainer):
        options = self.options
        sequence_length = trainer.num_features
        num_filters = options.num_filters
        emb_dim = options.emb_dim
        filter_sizes = options.filter_sizes

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        #dropout_keep_prob = tf.Variable(tf.float32, name="dropout_keep_prob")

        # Embedding layer
        #with tf.device('/cpu:0'), tf.name_scope("embedding"):
        init_width = 0.5 / emb_dim
        W = tf.Variable(
            #tf.random_uniform([trainer.total_features, emb_dim], -1.0, 1.0),
            tf.random_uniform([trainer.total_features, emb_dim], -init_width, init_width),
            name="W")
        embedded_chars = tf.nn.embedding_lookup(W, trainer.X)
        #[sentence_length, emb_dim, 1]
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

        #embedded_chars2 = tf.reduce_mean(embedded_chars, 1)
        #w_o = melt.init_weights([emb_dim, 1], name = 'w_o') # create symbolic variables
        #b_o = melt.init_bias([1], name = 'b_o')  
        #return tf.matmul(embedded_chars2, w_o) 
        
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, emb_dim, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                #W = tf.Variable(tf.random_normal(filter_shape, stddev=0.01), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")  
                conv = tf.nn.conv2d(
                    embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                #h = tf.nn.sigmoid(tf.nn.bias_add(conv, b), name="sigmoid")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    #[batch, height, width, channels]
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        #pooled_outputs [batch_size, 1, 1, num_filters]
        #h_pool [batch_size, 1, ,1, num_filters_total]
        h_pool = tf.concat(3, pooled_outputs)
        #we combine them into one long feature vector of shape [batch_size, num_filters_total]. 
        #Using -1 in tf.reshape tells TensorFlow to flatten the dimension when possible
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # Add dropout
        h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
        #dropout_keep_prob = tf.Variable(tf.constant(options.dropout_keep_prob), name="b")
        #h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        W = tf.Variable(tf.truncated_normal([num_filters_total, NUM_CLASSES], stddev=0.1), name="W")
        #W = tf.Variable(tf.random_normal([num_filters_total, 1], stddev=0.01), name="W")
        b = tf.Variable(tf.constant([0.1] * NUM_CLASSES, shape=[NUM_CLASSES]), name="b")

        #l2_loss += tf.nn.l2_loss(W)
        #l2_loss += tf.nn.l2_loss(b)
        return tf.nn.xw_plus_b(h_drop, W, b, name="score")
        #return tf.matmul(h_drop, W) + b

    def gen_feed_dict(self, trainer, trX, trY, test_mode = False):
        dict_ = trainer.gen_feed_dict(trX, trY)
        if not test_mode:
            dict_[self.dropout_keep_prob] = self.options.dropout_keep_prob
        else:
            dict_[self.dropout_keep_prob] = 1.0
        return dict_


    def __getstate__(self):
        d = dict(self.__dict__)
        del d['dropout_keep_prob']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d) # I *think* this is a safe way to do it

 

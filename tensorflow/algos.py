#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   algos.py
#        \author   chenghuige  
#          \date   2015-12-28 20:03:10.001711
#   \Description  
# ==============================================================================

import tensorflow as tf
import melt

#--------------------------------------------logistic regression 
class LogisticRegressionOptions(object):
    def __init__(self):
        self.type = 'logistic'
    
class LogisticRegression(object):
    def __init__(self, options = LogisticRegressionOptions()):
        self.options = options
        self.type = 'logistic'
        #self.weight = None

    def model(self, X, w, b):
        return melt.matmul(X,w) + b
    
    def forward(self, trainer):
        w = melt.init_weights([trainer.num_features, 1], name = 'w') 
        b = melt.init_bias([1], name = 'b')
        py_x = self.model(trainer.X, w, b)
        return py_x

#------------------------------------------one hidden layer mlp
class MlpOptions(object):
    def __init__(self):
        self.type = 'mlp'
        self.activation = 'sigmoid'
        self.hidden_size = 10
    
class Mlp(object):
    def __init__(self, options = MlpOptions()):
        self.options = options
        self.type = 'mlp'
        self.activation = melt.activation_map[options.activation]
        self.hidden_size = options.hidden_size
        #self.weight = None

    def model(self, X, w_h, b_h, w_o, b_o):
        h = self.activation(melt.matmul(X, w_h) + b_h) # this is a basic mlp, think 2 stacked logistic regressions
        return tf.matmul(h, w_o) + b_o # note that we dont take the softmax at the end because our cost fn does that for us
    
    def forward(self, trainer):
        w_h = melt.init_weights([trainer.num_features, self.hidden_size], name = 'w_h') # create symbolic variables
        b_h = melt.init_bias([1], name = 'b_h')        
        w_o = melt.init_weights([self.hidden_size, 1], name = 'w_o')
        #self.weight = w_o #if add this can not cpickle dump
        b_o = melt.init_bias([1], name = 'b_o')
        py_x = self.model(trainer.X, w_h, b_h, w_o, b_o)
        return py_x  
        #return py_x, w_o

#-----------------------------------------------------Mlp2 for 2 hidden layer
class Mlp2Options(object):
    def __init__(self):
        self.type = 'mlp2'
        self.activation = 'sigmoid'
        self.activation2 = 'sigmoid'
        self.hidden_size = 10
        self.hidden2_size = 5
    
class Mlp2(object):
    def __init__(self, options = Mlp2Options()):
        self.options = options
        self.type = 'mlp2'
        self.activation = melt.activation_map[options.activation]
        self.activation2 = melt.activation_map[options.activation2]
        self.hidden_size = options.hidden_size
        self.hidden2_size = options.hidden2_size
        #self.weight = None

    def model(self, X, w_h, b_h, w_h2, b_h2, w_o, b_o):
        h = self.activation(melt.matmul(X, w_h) + b_h) # this is a basic mlp, think 2 stacked logistic regressions
        h2 = self.activation(tf.matmul(h, w_h2) + b_h2)
        return tf.matmul(h2, w_o) + b_o # note that we dont take the softmax at the end because our cost fn does that for us
    
    def forward(self, trainer):
        w_h = melt.init_weights([trainer.num_features, self.hidden_size], name = 'w_h') # create symbolic variables
        b_h = melt.init_bias([1], name = 'b_h')       
        w_h2 = melt.init_weights([self.hidden_size, self.hidden2_size], name = 'w_h2')
        b_h2 = melt.init_bias([1], name = 'b_h2') 
        w_o = melt.init_weights([self.hidden2_size, 1], name = 'w_o')
        #self.weight = w_o #if add this can not cpickle dump
        b_o = melt.init_bias([1], name = 'b_o')
        py_x = self.model(trainer.X, w_h, b_h, w_h2, b_h2, w_o, b_o)
        return py_x  
        #return py_x, w_o


#------------------------------------------------ 1gram model using word embedding
#cbow for each word will assign an embedding vector, cbow is for text processing, so assume input is sparse input only
#also use 1gram version feature file
class CBOWOptions(object):
    def __init__(self):
        self.type = 'cbow'
        self.activation = 'sigmoid'
        self.emb_dim = 128

class CBOW(object):
    def __init__(self, options = CBOWOptions()):
        self.options = options
        self.type = 'cbow'
        self.activation = melt.activation_map[options.activation]

    def forward(self, trainer):
        opts = self.options
        init_width = 0.5 / opts.emb_dim
        vocab_size = trainer.num_features

        emb = tf.Variable(
            tf.random_uniform(
                [vocab_size, opts.emb_dim], -init_width, init_width),
            name="emb")

        w_o = melt.init_weights([opts.emb_dim, 1], name = 'w_o') # create symbolic variables
        b_o = melt.init_bias([1], name = 'b_o')  
 
        text_emb = tf.nn.embedding_lookup_sparse(emb, trainer.sp_ids, sp_weights = None, name = 'text_emb')

        #return tf.matmul(self.activation(text_emb), w_o) + b_o
        return tf.matmul(text_emb, w_o) + b_o



#--------------------------------------------- simple cnn model, now the most simple one, padding to make each post with the same length
class CnnOptions(object):
    def __init__(self):
        self.emb_dim = 128
        #self.emb_dim = 32
        
        #self.emb_dim = 128
        #self.emb_dim = 512

        #self.emb_dim = 256

        self.filter_sizes = [3, 4, 5]
        #self.filter_sizes = [5, 6, 7]
        #self.filter_sizes = [6, 7, 8]
        #self.filter_sizes = [10, 11, 12]
        #self.filter_sizes = [3, 5, 9]
        #self.filter_sizes = [3, 9, 15]
        #self.filter_sizes = [3, 7, 15]
        #self.filter_sizes = [3, 7, 5, 11, 15]
        #self.filter_sizes = [11, 15, 17]
        self.num_filters = 128
        #self.num_filters = 256
        #self.num_filters = 32
        
        self.dropout_keep_prob = 0.5
        #self.dropout_keep_prob = 0.8

        #self.dropout_keep_prob = 1.0

#@TODO @FIXME Algos save not using cpickle or find ways of cpickle dump without some attributes?
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
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(3, pooled_outputs)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # Add dropout
        h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
        #dropout_keep_prob = tf.Variable(tf.constant(options.dropout_keep_prob), name="b")
        #h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        W = tf.Variable(tf.truncated_normal([num_filters_total, 1], stddev=0.1), name="W")
        #W = tf.Variable(tf.random_normal([num_filters_total, 1], stddev=0.01), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[1]), name="b")

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



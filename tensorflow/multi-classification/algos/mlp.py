#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   mlp.py
#        \author   chenghuige  
#          \date   2016-02-24 16:00:51.881263
#   \Description  
# ==============================================================================

import tensorflow as tf
import melt
#------------------------------------------one hidden layer mlp
class MlpOptions(object):
    def __init__(self):
        self.type = 'mlp'
        self.activation = 'sigmoid'
        self.hidden_size = 10

from melt import NUM_CLASSES
from melt import NUM_FEATURES
#NUM_CLASSES = FLAGS.num_classes
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
        num_features = trainer.num_features if trainer.num_features > 0 else NUM_FEATURES
        with tf.device('/cpu:0'):
            w_h = melt.init_weights([num_features, self.hidden_size], name = 'w_h') # create symbolic variables
        b_h = melt.init_bias([self.hidden_size], name = 'b_h')    #@FIXME should be hidden_size ?    
        num_classes = trainer.num_classes if trainer.num_classes > 0 else NUM_CLASSES
        w_o = melt.init_weights([self.hidden_size, num_classes], name = 'w_o')
        #self.weight = w_o #if add this can not cpickle dump
        b_o = melt.init_bias([num_classes], name = 'b_o')
        py_x = self.model(trainer.X, w_h, b_h, w_o, b_o)

        #self.l1 = tf.reduce_sum(tf.abs(w_h)) + tf.reduce_sum(tf.abs(w_o))
        #self.l2 = tf.nn.l2_loss(w_h) + tf.nn.l2_loss(w_o)
        return py_x  
        #return py_x, tf.nn.l2_loss(w_h) + tf.nn.l2_loss(w_o)

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
        b_h = melt.init_bias([self.hidden_size], name = 'b_h')       
        w_h2 = melt.init_weights([self.hidden_size, self.hidden2_size], name = 'w_h2')
        b_h2 = melt.init_bias([self.hidden2_size], name = 'b_h2') 
        w_o = melt.init_weights([self.hidden2_size, NUM_CLASSES], name = 'w_o')
        #self.weight = w_o #if add this can not cpickle dump
        b_o = melt.init_bias([NUM_CLASSES], name = 'b_o')
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
        self.hidden_size = 10

# class CBOW(object):
#     def __init__(self, options = CBOWOptions()):
#         self.options = options
#         self.type = 'cbow'
#         self.activation = melt.activation_map[options.activation]

#     def forward(self, trainer):
#         opts = self.options
#         init_width = 0.5 / opts.emb_dim
#         vocab_size = trainer.num_features

#         emb = tf.Variable(
#             tf.random_uniform(
#                 [vocab_size, opts.emb_dim], -init_width, init_width),
#             name="emb")

#         w_o = melt.init_weights([opts.emb_dim, NUM_CLASSES], name = 'w_o') # create symbolic variables
#         b_o = melt.init_bias([NUM_CLASSES], name = 'b_o')  
 
#         text_emb = tf.nn.embedding_lookup_sparse(emb, trainer.sp_ids, sp_weights = None, name = 'text_emb')

#         #return tf.matmul(self.activation(text_emb), w_o) + b_o
#         return tf.matmul(text_emb, w_o) + b_o


class CBOW(object):
    def __init__(self, options = CBOWOptions()):
        self.options = options
        opts = self.options
        self.type = 'cbow'
        self.activation = melt.activation_map[options.activation]

        self.emb_dim = opts.emb_dim
        self.hidden_size = opts.hidden_size

    def forward(self, trainer):
        opts = self.options
        init_width = 0.5 / opts.emb_dim
        vocab_size = trainer.num_features

        emb = tf.Variable(
            tf.random_uniform(
                [vocab_size, opts.emb_dim], -init_width, init_width),
            name="emb")

        w_h = melt.init_weights([self.emb_dim, self.hidden_size], name = 'w_h') # create symbolic variables
        b_h = melt.init_bias([self.hidden_size], name = 'b_h')  
        w_o = melt.init_weights([self.hidden_size, NUM_CLASSES], name = 'w_o') # create symbolic variables
        b_o = melt.init_bias([NUM_CLASSES], name = 'b_o')  
 
        text_emb = tf.nn.embedding_lookup_sparse(emb, trainer.sp_ids, sp_weights = None, name = 'text_emb')

        h = self.activation(melt.matmul(text_emb, w_h) + b_h) # this is a basic mlp, think 2 stacked logistic regressions
        return tf.matmul(h, w_o) + b_o # note that we dont take the softmax at the end because our cost fn does that for us
 

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


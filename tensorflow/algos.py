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

 
class LogisticRegressionOptions(object):
    def __init__(self):
        self.type = 'logistic'
    
class LogisticRegression(object):
    def __init__(self, options = LogisticRegressionOptions()):
        self.options = options
        self.type = 'logistic'
    def model(self, X, w, b):
        return melt.matmul(X,w) + b
    
    def forward(self, trainer):
        w = melt.init_weights([trainer.num_features, 1], name = 'w') 
        b = melt.init_bias([1], name = 'b')
        py_x = self.model(trainer.X, w, b)
        return py_x

class MlpOptions(object):
    def __init__(self):
        self.type = 'mlp'
        self.activation = 'sigmoid'
        self.hidden_size = 10
    
class Mlp(object):
    def __init__(self, options = MlpOptions()):
        self.options = options
        self.type = 'mlp'
        self.activation = tf.nn.sigmoid
        self.hidden_size = options.hidden_size
        if options.activation == 'tanh':
            self.activation = tf.nn.tanh
    def model(self, X, w_h, b_h, w_o, b_o):
        h = self.activation(melt.matmul(X, w_h) + b_h) # this is a basic mlp, think 2 stacked logistic regressions
        return tf.matmul(h, w_o) + b_o # note that we dont take the softmax at the end because our cost fn does that for us
    
    def forward(self, trainer):
        w_h = melt.init_weights([trainer.num_features, self.hidden_size], name = 'w_h') # create symbolic variables
        b_h = melt.init_bias([1], name = 'b_h')        
        w_o = melt.init_weights([self.hidden_size, 1], name = 'w_o')
        b_o = melt.init_bias([1], name = 'b_o')
        py_x = self.model(trainer.X, w_h, b_h, w_o, b_o)
        return py_x    
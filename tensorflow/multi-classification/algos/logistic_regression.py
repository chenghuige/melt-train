#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   logistic_regression.py
#        \author   chenghuige  
#          \date   2016-02-24 16:02:54.619748
#   \Description  
# ==============================================================================

import tensorflow as tf
import melt
from melt import NUM_CLASSES
#NUM_CLASSES = FLAGS.num_classes
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
        w = melt.init_weights([trainer.num_features, NUM_CLASSES], name = 'w') 
        b = melt.init_bias([NUM_CLASSES], name = 'b')
        py_x = self.model(trainer.X, w, b)
        return py_x

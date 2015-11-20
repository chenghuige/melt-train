#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   slice.py
#        \author   chenghuige  
#          \date   2015-11-19 14:07:39.734096
#   \Description  
# ==============================================================================

import tensorflow as tf
import numpy as np

def model(X, w):
    return tf.matmul(X, w) # notice we use the same model as linear regression, this is because there is a baked in cost function which performs softmax and cross entropy

X = tf.placeholder("float", [2, 3]) # create symbolic variables
W = tf.placeholder("float", [3, 1])
#W = tf.placeholder("float", [3,])


x  = np.array([[1,2,3], [4,5,6]], dtype = np.float32)
print x.shape

w = np.array([[1],[2],[3]], dtype=np.float32)
#w = np.transpose(np.array([1, 2, 3], dtype=np.float32))
print w.shape 

py_x = model(X, W)

sess = tf.Session()
print sess.run(py_x, feed_dict={X: x, W: w})


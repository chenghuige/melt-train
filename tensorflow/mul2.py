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
    #return tf.matmul(X, tf.transpose(w)) # notice we use the same model as linear regression, this is because there is a baked in cost function which performs softmax and cross entropy
    return tf.matmul(X, w, transpose_b = True) # notice we use the same model as linear regression, this is because there is a baked in cost function which performs softmax and cross entropy

X = tf.placeholder("float", [2, 3]) # create symbolic variables
W = tf.placeholder("float", [2, 3])

B = tf.constant([1.0,2.0])

B2 = tf.reshape(B, [2])

x  = np.array([[1,2,3], [4,5,6]], dtype = np.float32)
print x
w = np.array([[1,4,5], [3, 2, 1]], dtype=np.float32)
print w
py_x = model(X, W)

sess = tf.Session()
print sess.run(py_x, feed_dict={X: x, W: w})

py_x2 = model(X, W) + B 
print sess.run(py_x2, feed_dict={X: x, W: w})

py_x3 = model(X, W) + B2 
print sess.run(py_x3, feed_dict={X: x, W: w})

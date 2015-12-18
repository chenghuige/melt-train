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
    return tf.mul(X, w) # notice we use the same model as linear regression, this is because there is a baked in cost function which performs softmax and cross entropy

X = tf.placeholder("float", [2, 3]) # create symbolic variables
W = tf.placeholder("float", [2, 3])
B = tf.placeholder("float", [2,])
B2 = tf.placeholder("float", [2,1])


x  = np.array([[1,2,3], [4,5,6]], dtype = np.float32)
print x
w = np.array([[1,4,5], [3, 2, 1]], dtype=np.float32)
print w
py_x = model(X, W)

true_logits = tf.reduce_sum(py_x, 1)
true_logits2 = tf.reduce_sum(py_x, 0)
true_logits3 = tf.reduce_sum(py_x, 1) + B
true_logits4 = tf.reduce_sum(py_x, 1) + B2
sess = tf.Session()
print sess.run(py_x, feed_dict={X: x, W: w})
logits = sess.run(true_logits, feed_dict={X: x, W: w})
print logits 
print logits.shape
print sess.run(true_logits2, feed_dict={X: x, W: w})
print sess.run(true_logits3, feed_dict={X: x, W: w, B : [1, 2]})
print sess.run(true_logits4, feed_dict={X: x, W: w, B2 : [[1], [2]]})


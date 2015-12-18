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

X = tf.placeholder("float", [4, 3]) # create symbolic variables
B = tf.placeholder("float", [4, 1]) # create symbolic variables

x  = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype = np.float32)
b  = np.array([[1], [2], [3], [4]], dtype = np.float32)

m_b = tf.Variable(tf.zeros([4]), name="sm_b")


Examples = tf.placeholder("float", [2])

examples = [0, 1]

lookup = tf.nn.embedding_lookup(X, examples)
lookup2 = tf.nn.embedding_lookup(B, examples)

lookup3 = tf.nn.embedding_lookup(m_b, examples)
sampled_b_vec = tf.reshape(lookup3, [2])

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

result = sess.run(lookup, feed_dict={X: x, Examples : examples})
print result 
print result.shape

print sess.run(lookup2, feed_dict={B: b, Examples : examples})

print sess.run(lookup3, feed_dict = {Examples : examples})

print sess.run(sampled_b_vec, feed_dict = {Examples : examples})
print sess.run(sampled_b_vec, feed_dict = {Examples : examples}).shape

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


Examples = tf.placeholder("int64", [2])

examples = [0, 1]

examples2 = [[0,1], [2, 1]]

lookup = tf.nn.embedding_lookup(X, Examples)
lookup2 = tf.nn.embedding_lookup(B, examples)

lookup3 = tf.nn.embedding_lookup(m_b, examples)
sampled_b_vec = tf.reshape(lookup3, [2])

lookup_mean = tf.reduce_mean(lookup, 0)


lookup_2example = tf.nn.embedding_lookup(x, examples2)
lookup_2example_mean = tf.reduce_mean(lookup_2example, 1)

#lookup_sparse = tf.nn.embedding_lookup_sparse(x, examples2, None)


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

result, mean_ = sess.run([lookup, lookup_mean], feed_dict={X: x, Examples : examples})
print result 
print result.shape

print 'mean is: ', mean_ 

print 'lookup2 results:'
print sess.run(lookup2, feed_dict={B: b, Examples : examples})

print sess.run(lookup3, feed_dict = {Examples : examples})

print sess.run(sampled_b_vec, feed_dict = {Examples : examples})
print sess.run(sampled_b_vec, feed_dict = {Examples : examples}).shape


print 'lookup 2example'
print sess.run(lookup_2example)
print 'lookup_2example_mean'
print sess.run(lookup_2example_mean)
#print 'lookup sparse mean'
#print sess.run(lookup_sparse)


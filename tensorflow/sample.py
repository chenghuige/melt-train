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

X = tf.placeholder("float", [2, 3]) # create symbolic variables
X2 = tf.placeholder("float", [2, 3])
B = tf.placeholder("float", [2, 1])

x  = np.array([[1,2,3], [4,5,6]], dtype = np.float32)
print x

x2  = np.array([[2,1,1], [3,5,4]], dtype = np.float32)
print x2

b = np.array([[1],[2]], dtype=np.float32)
print b

sampled_b_vec = tf.reshape(B, [2])

sampled_logits_nobias = tf.matmul(X, X2, transpose_b=True)

sampled_logits = tf.matmul(X, X2, transpose_b=True) + sampled_b_vec

sess = tf.Session()
print sess.run(sampled_b_vec, feed_dict={X: x, X2: x2, B: b})

print sess.run(sampled_logits_nobias, feed_dict={X: x, X2: x2, B: b})

print sess.run(sampled_logits, feed_dict={X: x, X2: x2, B: b})
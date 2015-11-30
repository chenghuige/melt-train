#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   sparse_tensor.py
#        \author   chenghuige  
#          \date   2015-11-29 11:48:19.860197
#   \Description  
# ==============================================================================

#Suppose you have a minibatch of 2 entries. 
#The first entry has sparse ids [53, 87, 101], values [0.1, 0.2, 0.3] 
#and the second has sparse ids [34, 98], weights [-1.0, 3.5]. 
#Suppose your total vocab size is 500. Suppose also that the hidden layer has depth 25 (25 units).
#y_values should be the output of X*[w1, w2], where w1 and w2 are the two minibatch entries.
import tensorflow as tf
import numpy as np

X = tf.placeholder("float", [10, 1])
x = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]], dtype=np.float32)

sp_indices = tf.placeholder(tf.int64)
sp_shape = tf.placeholder(tf.int64)
sp_ids_val = tf.placeholder(tf.int64)
sp_weights_val = tf.placeholder(tf.float32)
sp_ids = tf.SparseTensor(sp_indices, sp_ids_val, sp_shape)
sp_weights = tf.SparseTensor(sp_indices, sp_weights_val, sp_shape)
y = tf.nn.embedding_lookup_sparse(X, sp_ids, sp_weights, combiner = "sum")

sess = tf.Session()
sess.run(tf.initialize_all_variables())

y_values = sess.run(y, feed_dict={
  X: x,
  sp_indices: [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]],  # 3 entries in minibatch entry 0, 2 entries in entry 1.
  sp_shape: [2, 3],  # batch size: 2, max index: 2 (so index count == 3)
  sp_ids_val: [2, 5, 8, 3, 4],
  sp_weights_val: [1.0, 1.5, 2.5, 3.5, 4.5]
  })

print y_values

#0 should be
# 2 * 1 + 5 * 1.5 + 8 * 2.5  = 29.5

y_values = sess.run(y, feed_dict={
  X: x,
  sp_indices: [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]],  # 3 entries in minibatch entry 0, 2 entries in entry 1.
  sp_shape: [2, 3],  # batch size: 2, max index: 2 (so index count == 3)
  sp_ids_val: [2, 5, 8, 3, 4],
  sp_weights_val: [3.0, 1.5, 2.5, 3.5, 4.5]
  })

print y_values

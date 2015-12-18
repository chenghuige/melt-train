#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   zero_out.py
#        \author   chenghuige  
#          \date   2015-12-05 19:09:24.645282
#   \Description  
# ==============================================================================

import tensorflow as tf

input_a = tf.constant([[1, 2]])
#input_a = tf.constant([1, 2])
input_b = tf.constant([[1],[2]])

output = tf.matmul(input_a, input_b)
#output = tf.matmul(input_a, input_a, transpose_b=True)

with tf.Session() as sess:
	init = tf.initialize_all_variables()
	sess.run(init)
	print output.eval()
	print output.eval().shape


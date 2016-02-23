#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   zero_out.py
#        \author   chenghuige  
#          \date   2015-12-05 19:09:24.645282
#   \Description  
# ==============================================================================

import tensorflow as tf

input = tf.constant([[1, 2, 3], [3, 4, 5]])

zero_out_module = tf.load_op_library('zero_out.so')
#output = tf.user_ops.zero_out(input)
output = zero_out_module.zero_out(input)

with tf.Session() as sess:
	init = tf.initialize_all_variables()
	sess.run(init)
	print 'before zero out ', input.eval()
	print 'after zero out', output.eval()


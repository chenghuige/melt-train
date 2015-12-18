#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   zero_out.py
#        \author   chenghuige  
#          \date   2015-12-05 19:09:24.645282
#   \Description  
# ==============================================================================

import tensorflow as tf

input = tf.constant([[6, 2, 3], [3, 8, 5]])

output = tf.arg_max(input[1,:], 0)

with tf.Session() as sess:
	init = tf.initialize_all_variables()
	sess.run(init)
	print output.eval()
	print output.eval().shape
 

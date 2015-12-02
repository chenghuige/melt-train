#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   save.py
#        \author   chenghuige  
#          \date   2015-12-01 12:39:30.959606
#   \Description  
# ==============================================================================


import tensorflow as tf
import numpy as np
with tf.Session() as sess:
	a = tf.Variable(5.0, name='a')
	b = tf.Variable(6.0, name='b')
	c = tf.mul(a, b, name="c")
	
	sess.run(tf.initialize_all_variables())
	
	print a.eval() # 5.0
	print b.eval() # 6.0
	print c.eval() # 30.0

	tf.train.write_graph(sess.graph_def, 'models/', 'train.pb', as_text=True)

 

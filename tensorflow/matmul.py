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
W = tf.placeholder("float", [3, 1])
py_x = tf.matmul(X, W)

x  = np.array([[1,2,3], [4,5,6]], dtype = np.float32)
w = np.array([[1],[2],[3]], dtype=np.float32)

sess = tf.Session()
print sess.run(py_x, feed_dict={X: x, W: w})


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

a = tf.placeholder("float", [2, 3]) # create symbolic variables
b = tf.placeholder("float") # Create a symbolic variable 'b'

y = tf.mul(a, b)
z = tf.slice(a, [0, 1], [2, 1])

sess = tf.Session()
#init = tf.initialize_all_variables()
#sess.run(init)

a_ = [[1,2,3], [4,5,6]]
print sess.run(y, feed_dict={a: a_, b: 2})
print sess.run(z, feed_dict={a: a_})


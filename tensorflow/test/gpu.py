#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   gpu.py
#        \author   chenghuige  
#          \date   2016-02-16 19:44:11.869411
#   \Description  
# ==============================================================================
import tensorflow as tf
# Creates a graph.
# with tf.device('/gpu:0'):
# 	a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
# 	b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
# 	c = tf.matmul(a, b)
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print sess.run(c)


 

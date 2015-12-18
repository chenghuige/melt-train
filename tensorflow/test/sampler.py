#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   sampler.py
#        \author   chenghuige  
#          \date   2015-12-03 00:23:54.649751
#   \Description  
# ==============================================================================

import tensorflow as tf

vocab = range(99, -1, -1)
labels = [0, 1, 3, 8, 22, 32, 15, 2, 5, 4]

Vocab = tf.placeholder(tf.int64, [100])
Labels = tf.placeholder(tf.int64, [10])
labels_matrix = tf.reshape(
        tf.cast(Labels,
                dtype=tf.int64),
        [10, 1])


sampled_ids, a, b = (tf.nn.fixed_unigram_candidate_sampler(
    true_classes=labels_matrix,
    num_true=1,
    num_sampled=4,
    unique=True,
    range_max=100,
    distortion=1,
    unigrams=vocab)) #notice unigrams is a list not tensor
    
sess = tf.Session() 
init = tf.initialize_all_variables()
sess.run(init)

print sess.run(sampled_ids, feed_dict = {Labels : labels})

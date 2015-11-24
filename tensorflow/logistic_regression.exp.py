#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   logistic_regression.py
#        \author   chenghuige  
#          \date   2015-11-19 16:06:52.693026
#   \Description  
# ==============================================================================


import tensorflow as tf
import numpy as np
import melt_dataset
import sys
from sklearn.metrics import roc_auc_score

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w):
    return 1.0/(1.0 + tf.exp(-(tf.matmul(X, w))))

#./logistic_regression.py corpus/feature.normed.rand.12000.0_2.txt corpus/feature.normed.rand.12000.1_2.txt
#notice if setting batch_size too big here 500 will result in learning turn output nan if using learning_rate 0.01,
#to solve this large batch size need low learning rate 0.001 will be ok
batch_size = 500
learning_rate = 0.001
#batch_size = 500
#learning_rate = 0.13
num_iters = 100

argv = sys.argv 
trainset = argv[1]
testset = argv[2]

trX, trY = melt_dataset.load_dense_data(trainset)
print "finish loading train set ",trainset
teX, teY = melt_dataset.load_dense_data(testset)
print "finish loading test set ", testset

num_features = trX[0].shape[0]
print 'num_features: ',num_features 
print 'trainSet size: ', len(trX)
print 'testSet size: ', len(teX)
print 'batch_size:', batch_size, ' learning_rate:', learning_rate, ' num_iters:', num_iters

X = tf.placeholder("float", [None, num_features]) # create symbolic variables
Y = tf.placeholder("float", [None, 1])

w = init_weights([num_features, 1]) # like in linear regression, we need a shared variable weight matrix for logistic regression

py_x = model(X, w)

cost = -tf.reduce_sum(Y*tf.log(py_x) + (1 - Y) * tf.log(1 - py_x))
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) # construct optimizer

predict_op = py_x

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in range(num_iters):
    predicts, cost_ = sess.run([predict_op, cost], feed_dict={X: teX, Y: teY})
    print i, 'auc:', roc_auc_score(teY, predicts), 'cost:', cost_
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
			sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

predicts, cost_ = sess.run([predict_op, cost], feed_dict={X: teX, Y: teY})
print 'final ', 'auc:', roc_auc_score(teY, predicts),'cost:', cost_

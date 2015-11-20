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
    #return tf.matmul(X, w) # notice we use the same model as linear regression, this is because there is a baked in cost function which performs softmax and cross entropy
    return 1.0/(1.0 + tf.exp(-(tf.matmul(X, w))))
    #return tf.nn.sigmoid((tf.matmul(X, w)))


#def model(X, w):
#    return tf.matmul(X, w) # notice we use the same model as linear regression, this is because there is a baked in cost function which performs softmax

batch_size = 10
learning_rate = 0.01

argv = sys.argv 
trainset = argv[1]
testset = argv[2]

trX, trY = melt_dataset.load_dense_data(trainset)
print "finish loading train set ",trainset
teX, teY = melt_dataset.load_dense_data(testset)
print "finish loading test set ", testset

num_features = trX[0].shape[0]
print 'num_features: ',num_features

X = tf.placeholder("float", [None, num_features]) # create symbolic variables
Y = tf.placeholder("float", [None, 1])
#Y = tf.placeholder("float", [None, 2])

w = init_weights([num_features, 1]) # like in linear regression, we need a shared variable weight matrix for logistic regression
#w = init_weights([num_features, 2]) # like in linear regression, we need a shared variable weight matrix for logistic regression

py_x = model(X, w)

cost = -tf.reduce_sum(Y*tf.log(py_x) + (1 - Y) * tf.log(1 - py_x))
cost1 = py_x
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y)) # compute mean cross entropy (softmax is applied internally)
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) # construct optimizer


predict_op = py_x
#predict_op =  tf.nn.sigmoid(py_x)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

out = open('predict.txt', 'w')
for i in range(100):
    predicts = sess.run(predict_op, feed_dict={X: teX, Y: teY})
    print roc_auc_score(teY, predicts)
    #print sess.run(cost1, feed_dict={X: teX, Y: teY})
    print sess.run(cost, feed_dict={X: teX, Y: teY})
    #print predicts
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
			#print trX[start:end]
			sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
    #print i, np.mean(sess.run(predict_op, feed_dict={X: teX, Y: teY}))
    #predicts = sess.run(predict_op, feed_dict={X: teX, Y: teY})
    #print predicts
    #if i == 99:
    #    for j in xrange(len(predicts)):
    #        #out.write('%f\t%f\n'%(teY[j][1], predicts[j][1]))
    #        out.write('%f\t%f\n'%(teY[j], predicts[j]))
    #for j in xrange(len(predicts)): 
    #    print teY[j][1], ' ', predicts[j][1]
    #print roc_auc_score(teY[:,1], predicts[:,1])
    #print roc_auc_score(teY, predicts)
 

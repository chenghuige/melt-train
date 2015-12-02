#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   binary_classification.py
#        \author   chenghuige  
#          \date   2015-11-30 16:06:52.693026
#   \Description  
# ==============================================================================

import sys

import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score

import melt

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 120, 'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size', 500, 'Batch size. Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train', './corpus/feature.normed.rand.12000.0_2.txt', 'train file')
flags.DEFINE_string('test', './corpus/feature.normed.rand.12000.1_2.txt', 'test file')
flags.DEFINE_string('method', 'logistic', 'currently support logistic/mlp')
#----for mlp
flags.DEFINE_integer('hidden_size', 20, 'Hidden unit size')

trainset_file = FLAGS.train
testset_file = FLAGS.test

learning_rate = FLAGS.learning_rate 
num_epochs = FLAGS.num_epochs 
batch_size = FLAGS.batch_size 

method = FLAGS.method

trainset = melt.load_dataset(trainset_file)
print "finish loading train set ",trainset_file
testset = melt.load_dataset(testset_file)
print "finish loading test set ", testset_file

assert(trainset.num_features == testset.num_features)
num_features = trainset.num_features
print 'num_features: ', num_features
print 'trainSet size: ', trainset.num_instances()
print 'testSet size: ', testset.num_instances()
print 'batch_size:', batch_size, ' learning_rate:', learning_rate, ' num_epochs:', num_epochs


trainer = melt.gen_binary_classification_trainer(trainset)

class LogisticRegresssion:
	def model(self, X, w, b):
		return melt.matmul(X,w) + b
	
	def forward(self, trainer):
		self.w = melt.init_weights([trainer.num_features, 1]) 
		self.b = melt.init_bias([1])
		py_x = self.model(trainer.X, self.w, self.b)
		return py_x

class Mlp:
	def model(self, X, w_h, w_o, b):
		h = tf.nn.sigmoid(melt.matmul(X, w_h) + b) # this is a basic mlp, think 2 stacked logistic regressions
		return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us
	
	def forward(self, trainer):
		self.w_h = melt.init_weights([trainer.num_features, FLAGS.hidden_size]) # create symbolic variables
		self.w_o = melt.init_weights([FLAGS.hidden_size, 1])
		self.b = melt.init_bias([1])

		py_x = self.model(trainer.X, self.w_h, self.w_o, self.b)
		return py_x	

def gen_algo(method):
	if method == 'logistic':
		return LogisticRegresssion()
	elif method == 'mlp':
		return Mlp()
	else:
		print method, ' is not supported right now'
		exit(-1)

algo = gen_algo(method)
py_x = algo.forward(trainer)
Y = trainer.Y

cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) # construct optimizer
predict_op = tf.nn.sigmoid(py_x)

a = tf.constant(5, name='alpha')

sess = tf.Session() 
init = tf.initialize_all_variables()
sess.run(init)

summary_writer = tf.train.SummaryWriter('./summary', graph_def=sess.graph_def)

teX, teY = testset.full_batch()
num_train_instances = trainset.num_instances()
for i in range(num_epochs):
	predicts, cost_ = sess.run([predict_op, cost], feed_dict = trainer.gen_feed_dict(teX, teY))
	print i, 'auc:', roc_auc_score(teY, predicts), 'cost:', cost_ / len(teY)
	for start, end in zip(range(0, num_train_instances, batch_size), range(batch_size, num_train_instances, batch_size)):
		trX, trY = trainset.mini_batch(start, end)
		sess.run(train_op, feed_dict = trainer.gen_feed_dict(trX, trY))

predicts, cost_ = sess.run([predict_op, cost], feed_dict = trainer.gen_feed_dict(teX, teY))
print 'final ', 'auc:', roc_auc_score(teY, predicts),'cost:', cost_ / len(teY)

#tf.train.write_graph(sess.graph_def, 'models/', 'train.pb', as_text=True)
#print algo.w.eval(sess)
#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   binary_classification.py
#        \author   chenghuige  
#          \date   2015-11-30 16:06:52.693026
#   \Description  
# ==============================================================================

import sys, os

import tensorflow as tf
import numpy as np
#from sklearn.metrics import roc_auc_score
import cPickle

import melt

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 120, 'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size', 500, 'Batch size. Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train', './corpus/feature.normed.rand.12000.0_2.txt', 'train file')
flags.DEFINE_string('test', './corpus/feature.normed.rand.12000.1_2.txt', 'test file')
flags.DEFINE_string('method', 'mlp', 'currently support logistic/mlp')
flags.DEFINE_string('activation', 'sigmoid', 'you may try tanh or other activate function')
#----for mlp
flags.DEFINE_integer('hidden_size', 20, 'Hidden unit size')

#first train then test
flags.DEFINE_string('model', './model', 'model path')
flags.DEFINE_string('command', 'train', 'train or test')

trainset_file = FLAGS.train
testset_file = FLAGS.test

learning_rate = FLAGS.learning_rate 
num_epochs = FLAGS.num_epochs 
batch_size = FLAGS.batch_size 

method = FLAGS.method


print 'batch_size:', batch_size, ' learning_rate:', learning_rate, ' num_epochs:', num_epochs


class LogisticRegresssion(object):
    def model(self, X, w, b):
        return melt.matmul(X,w) + b
    
    def forward(self, trainer):
        w = melt.init_weights([trainer.num_features, 1], name = 'w') 
        b = melt.init_bias([1], name = 'b')
        py_x = self.model(trainer.X, w, b)
        return py_x

class Mlp(object):
    def __init__(self, activation = 'sigmoid', hidden_size = 10):
        self.activation = tf.nn.sigmoid
        self.hidden_size = hidden_size
        if activation == 'tanh':
            self.activation = tf.nn.tanh
    def model(self, X, w_h, b_h, w_o, b_o):
        h = self.activation(melt.matmul(X, w_h) + b_h) # this is a basic mlp, think 2 stacked logistic regressions
        return tf.matmul(h, w_o) + b_o # note that we dont take the softmax at the end because our cost fn does that for us
    
    def forward(self, trainer):
        w_h = melt.init_weights([trainer.num_features, self.hidden_size], name = 'w_h') # create symbolic variables
        b_h = melt.init_bias([1], name = 'b_h')        
        w_o = melt.init_weights([FLAGS.hidden_size, 1], name = 'w_o')
        b_o = melt.init_bias([1], name = 'b_o')
        py_x = self.model(trainer.X, w_h, b_h, w_o, b_o)
        return py_x    


class BinaryClassification(object):
    def gen_algo(self, method):
        self.method = method
        if method == 'logistic':
            return LogisticRegresssion()
        elif method == 'mlp':
            return Mlp(activation=FLAGS.activation, hidden_size=FLAGS.hidden_size)
        else:
            print method, ' is not supported right now'
            method = 'mlp'

    def build_graph(self, algo, trainer):
        py_x = algo.forward(trainer)
        Y = trainer.Y
        cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(py_x, Y))
        predict_op = tf.nn.sigmoid(py_x) 
        evaluate_op = tf.user_ops.auc(py_x, Y)

        self.cost = cost
        self.predict_op = predict_op
        self.evaluate_op = evaluate_op  
        return cost, predict_op, evaluate_op
        
    def foward(self, algo, trainer, learning_rate):
        cost, predict_op, evaluate_op = self.build_graph(algo, trainer)
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)   
        
        return cost, train_op, predict_op, evaluate_op

    def train(self, trainset_file, testset_file, method, num_epochs, learning_rate, model_path):
        trainset = melt.load_dataset(trainset_file)
        print "finish loading train set ", trainset_file
        self.num_features = trainset.num_features
        print 'num_features: ', self.num_features
        print 'trainSet size: ', trainset.num_instances()
        testset = melt.load_dataset(testset_file)
        print "finish loading test set ", testset_file
        assert(trainset.num_features == testset.num_features)
        print 'testSet size: ', testset.num_instances()

        algo = self.gen_algo(method)
        trainer = melt.gen_binary_classification_trainer(trainset)
        self.algo = algo
        self.trainer = trainer
        
        cost, train_op, predict_op, evaluate_op = self.foward(algo, trainer, learning_rate)
        
        self.session = tf.Session() 
        init = tf.initialize_all_variables()
        self.session.run(init)
        
        tf.scalar_summary("cross_entropy", cost)
        tf.scalar_summary("auc", evaluate_op)
        merged_summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter('/home/users/chenghuige/tmp/tensorflow_logs', self.session.graph_def)
        
        teX, teY = testset.full_batch()
        
        os.system('mkdir -p ' + FLAGS.model)
        num_train_instances = trainset.num_instances()
        for epoch in range(num_epochs):
            for start, end in zip(range(0, num_train_instances, batch_size), range(batch_size, num_train_instances, batch_size)):
                trX, trY = trainset.mini_batch(start, end)
                self.session.run(train_op, feed_dict = trainer.gen_feed_dict(trX, trY))
                #predicts, cost_ = sess.run([predict_op, cost], feed_dict = trainer.gen_feed_dict(teX, teY))
            #print epoch, ' auc:', roc_auc_score(teY, predicts),'cost:', cost_ / len(teY)
            predicts, auc_, cost_ = self.session.run([predict_op, evaluate_op, cost], feed_dict = trainer.gen_feed_dict(teX, teY))
            print epoch, ' auc:', auc_,'cost:', cost_ / len(teY)
            self.save_model(model_path, epoch)
        
        summary_str = self.session.run(merged_summary_op, feed_dict = trainer.gen_feed_dict(teX, teY))
        summary_writer.add_summary(summary_str, epoch)
        
        self.save_others(model_path)
    
    def save_model(self, model_path, epoch):
        tf.train.Saver().save(self.session, model_path + '/model.ckpt', global_step = epoch)
    
    def save_others(self, model_path):
        file_ = open(model_path + '/algo.ckpt', 'w')
        cPickle.dump(self.algo, file_)
        
        file_ = open(model_path + '/trainer.ckpt', 'w')

        #@FIXME can't pickle module objects error
        #cPickle.dump(self.trainer, file_)
        
        file_.write('%s\t%d'%(self.trainer.type, self.trainer.num_features))

    #since trainer and not dump by cPickle, another way is to load trainset again, but load
    #only one line data, so the save/load will use almost same code as train for reload
    #though a bit more redundant work for reconsructing the graph
    def load(self, model_path):  
        algo_file = open(model_path + '/algo.ckpt')
        self.algo = cPickle.load(algo_file)   
        
        trainer_file = open(model_path + '/trainer.ckpt')
        type_, self.num_features = trainer_file.read().strip().split('\t')
        self.num_features = int(self.num_features)
        if type_ == 'dense':
            self.trainer = melt.BinaryClassificationTrainer(num_features=self.num_features)
        else:
            self.trainer = melt.SparseBinaryClassificationTrainer(num_features=self.num_features) 
        
        self.build_graph(self.algo, self.trainer)
        
        self.session = tf.Session() 

        tf.train.Saver().restore(self.session, model_path + "/model.ckpt")
        
    def test(self, testset_file):
        testset = melt.load_dataset(testset_file)
        print "finish loading test set ", testset_file
        assert(testset.num_features == self.num_features)   
        teX, teY = testset.full_batch()
        predicts, auc_, cost_ = self.session.run([self.predict_op, self.evaluate_op, self.cost], feed_dict = self.trainer.gen_feed_dict(teX, teY))
        print ' auc:', auc_,'cost:', cost_ / len(teY)
        
def main():
    bc = BinaryClassification()
    if FLAGS.command == 'train':    
        bc.train(trainset_file, testset_file, method, num_epochs, learning_rate, FLAGS.model)
    elif FLAGS.command == 'test':
        bc.load(FLAGS.model)
        bc.test(testset_file)

if __name__ == "__main__":
    main()

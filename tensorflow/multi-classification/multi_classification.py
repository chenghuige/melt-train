#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   binary_classification.py
#        \author   chenghuige  
#          \date   2015-11-30 16:06:52.693026
#   \Description  
# ==============================================================================



import os

import time
import numpy as np

import tensorflow as tf 
#import nowarning


try:
    print 'tf.verion:', tf.__version__
except Exception:
    print 'tf version unknown, might be an old verion'

flags = tf.app.flags
FLAGS = flags.FLAGS

#flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 120, 'Number of epochs to run trainer.')
flags.DEFINE_integer('num_classes', 2, 'Number of epochs to run trainer.')
flags.DEFINE_integer('save_epochs', 1, 'save epochs every save_epochs round')
flags.DEFINE_integer('batch_size', 64, 'Batch size. Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train', './dataset/train', 'train file')
flags.DEFINE_string('test', './dataset/test', 'test file')
flags.DEFINE_string('method', 'mlp', 'currently support logistic/mlp/mlp2')

flags.DEFINE_boolean('ada_grad', True, 'use ada grad')

flags.DEFINE_boolean('shuffle', True, 'shuffle dataset each epoch')

flags.DEFINE_boolean('is_record', False, 'is_record means pre parse as tf standard record')

#----for mlp
flags.DEFINE_integer('hidden_size', 200, 'Hidden unit size')
flags.DEFINE_integer('hidden2_size', 5, 'Hidden unit size of second hidden layer')

flags.DEFINE_string('activation', 'relu', 'you may try tanh or other activate function')
flags.DEFINE_string('activation2', 'sigmoid', 'you may try tanh or other activate function')

#----for cbow
flags.DEFINE_integer('emb_dim', 256, 'embedding dimension')

#first train then test
flags.DEFINE_string('model', './model', 'model path')
flags.DEFINE_string('command', 'train', 'train or test')

#functiona flags
flags.DEFINE_boolean('show_device', False, 'show device info or not')

flags.DEFINE_boolean('use_summary', True, 'use summary from showing graph')

flags.DEFINE_boolean('auto_stop', False, 'auto stop iteration and store the best result')
flags.DEFINE_float('min_improve', 0.001 , 'stop when improve less then min_improve')


trainset_file = FLAGS.train
testset_file = FLAGS.test

learning_rate = FLAGS.learning_rate 
num_epochs = FLAGS.num_epochs 
batch_size = FLAGS.batch_size 

method = FLAGS.method


import cPickle

import melt

cout = open('cout.txt', 'w')

from algo import *

class AlgosFactory(object):
    def gen_algo(self, method):
        if method == 'logistic' or method == 'lr':
            option = LogisticRegressionOptions()
            return LogisticRegression(option)
        elif method == 'mlp':
            option = MlpOptions()
            option.activation = FLAGS.activation
            option.hidden_size = FLAGS.hidden_size
            return Mlp(option)
        elif method == 'mlp2':
            option = Mlp2Options()
            option.activation = FLAGS.activation
            option.activation2 = FLAGS.activation2
            option.hidden_size = FLAGS.hidden_size
            option.hidden_size2 = FLAGS.hidden2_size
            return Mlp2(option)
        elif method == 'cbow':
            option = CBOWOptions()
            option.emb_dim = FLAGS.emb_dim
            return CBOW(option)
        elif method == 'cnn':
            option = CnnOptions()
            return Cnn(option)
        elif method == 'rnn':
            option = RnnOptions()
            return Rnn(option)
        elif method == 'charcnn':
            option = CharCnnOptions()
            return CharCnn(option)
        else:
            print method, ' is not supported right now, will use default method mlp'
            method = 'mlp'
            return self.gen_algo(method)

class MultiClassifier(object):
    def __init__(self):
      self.avg_accuracy = 0.
      self.saver = None

    def gen_algo(self, method):
        self.method = method
        return AlgosFactory().gen_algo(method)
    
    def gen_algo_from_option(self, option):
        if option.type == 'logistic':
            return LogisticRegression(option) 
        elif option.type == 'mlp':
            return Mlp(option) 
        else:
            return None

    def build_graph(self, algo, trainer):
        #with tf.device('/cpu:0'):
        py_x = algo.forward(trainer)
        #self.py_x = py_x
        #a = tf.Print(py_x, [py_x], message="This is a: ")

        Y = trainer.Y
        #Y = tf.to_int64(Y)
        cost = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(py_x, Y))
        
        predict_op = tf.nn.sigmoid(py_x) 
        #correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        correct_prediction = tf.nn.in_top_k(py_x, Y, 1)
        #correct_prediction = tf.constant([1.0]*NUM_CLASSES)
        
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #accuracy = cost

        self.cost = cost
        self.predict_op = predict_op
        self.accuracy = accuracy 
        return cost, predict_op, accuracy
        
    def foward(self, algo, trainer, learning_rate):
        #with tf.device('/cpu:0'):
        cost, predict_op, evaluate_op = self.build_graph(algo, trainer)

        train_op = None
        if not trainer.index_only:
            if not FLAGS.ada_grad:
                global_step = tf.Variable(0, trainable=False)
                starter_learning_rate = FLAGS.learning_rate
                decay_step = int(1000000 / FLAGS.batch_size)
                learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               decay_step, 0.99, staircase=True)
                train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)   
                print 'use nomral gradient decent optimizer'
            else:
                train_op = tf.train.AdagradOptimizer(learning_rate).minimize(cost)    #gpu will fail...
                print 'use adagrad optimizer'
        else:
            ##@TODO WHY  train_op = tf.train.AdagradOptimizer(learning_rate).minimize(cost) will fail...
            #global_step = tf.Variable(0, name="global_step", trainable=False)
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            grads_and_vars = optimizer.compute_gradients(cost)
            train_op = optimizer.apply_gradients(grads_and_vars)
        self.train_op = train_op
        return cost, train_op, predict_op, evaluate_op
        
    def train(self, trainset_file, testset_file, method, num_epochs, learning_rate, model_path):
        print 'batch_size:', batch_size, ' learning_rate:', learning_rate, ' num_epochs:', num_epochs
        print 'method:',  method

        trainset = melt.load_dataset(trainset_file, is_record=FLAGS.is_record)
        if FLAGS.is_record:
            trainset.build_read_graph(batch_size)
        print "finish loading train set ", trainset_file
        # self.num_features = trainset.num_features
        # print 'num_features: ', self.num_features
        # print 'trainSet size: ', trainset.num_instances()
        testset = melt.load_dataset(testset_file, is_record=FLAGS.is_record)
        if FLAGS.is_record:
            testset.build_read_graph(batch_size * 10)
        print "finish loading test set ", testset_file
        # assert(trainset.num_features == testset.num_features)
        # print 'testSet size: ', testset.num_instances()

        algo = self.gen_algo(method)
        trainer = melt.gen_classification_trainer(trainset)
        self.algo = algo
        self.trainer = trainer
        print 'trainer_type:', trainer.type
        print 'trainer_index_only:', trainer.index_only
        
        cost, train_op, predict_op, evaluate_op = self.foward(algo, trainer, learning_rate)
        #self.foward(algo, trainer, learning_rate)

        config = None
        if not FLAGS.show_device:
            config = tf.ConfigProto()
        else:
            config=tf.ConfigProto(log_device_placement=True)
        config.gpu_options.allocator_type = 'BFC'

        #self.session = tf.Session(config=config) 
        self.session = tf.InteractiveSession()
        init_op = tf.group(tf.initialize_all_variables(),
                   tf.initialize_local_variables())
        #init = tf.initialize_all_variables()
        self.session.run(init_op)
        
        summary_writer = None
        if FLAGS.use_summary:
            tf.scalar_summary("cross_entropy", self.cost)
            tf.scalar_summary("accuracy@1", self.accuracy)
            merged_summary_op = tf.merge_all_summaries()
            summary_writer = tf.train.SummaryWriter(FLAGS.model, self.session.graph)
        
        #os.system('rm -rf ' + FLAGS.model)
        os.system('mkdir -p ' + FLAGS.model)
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5)
        self.save_info(model_path)

        if not FLAGS.is_record:
            for epoch in range(num_epochs):
                if epoch > 0 and FLAGS.shuffle:
                    trainset = melt.load_dataset(trainset_file)

                self.train_(trainset, testset=testset, epoch=epoch)
                self.test_(testset, epoch=epoch)
                
                #need_stop = self.test_(testset, epoch = epoch)
                #if need_stop:
                #    print 'need stop as improve is smaller then %f'%FLAGS.min_improve
                #    break

                if epoch % FLAGS.save_epochs == 0 and not trainer.index_only:
                    self.save_model(model_path, epoch)
        else:
            # Start input enqueue threads.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.session, coord=coord)

            try:
                step = 0
                start_time = time.time()
                while not coord.should_stop():
                    #self.trainer.X, self.trainer.Y = trainset.next_batch(self.session)
                    _, cost_, accuracy_ = self.session.run([self.train_op, self.cost, self.accuracy]) 
                    #cost_, accuracy_ = self.session.run([self.cost, self.accuracy]) 
                    if step % 100 == 0:
                        end_time = time.time()
                        duration = end_time - start_time
                        start_time = end_time
                        print 'step:', step, 'train precision@1:', accuracy_,'cost:', cost_, 'duration:', duration 
                    if step % 1000 == 0:
                        pass
                    step += 1
            except tf.errors.OutOfRangeError:
                #print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
                pass
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()

            # Wait for threads to finish.
            coord.join(threads)
      
        self.save_model(model_path)

        sess.close()

    
    def train_(self, trainset, testset=None, epoch=0):
        num_train_instances = trainset.num_instances()
        round = 0
        start_time = time.time()
        while True:
            trX, trY = trainset.next_batch(batch_size)
            if trX is None:
                break
            feed_dict = melt.gen_feed_dict(self.trainer, self.algo, trX, trY)
            _, cost_, accuracy_ = self.session.run([self.train_op, self.cost, self.accuracy], feed_dict=feed_dict)  
            #py_x = self.session.run(self.py_x, feed_dict = feed_dict)
            #print py_x
            if round % 100 == 0:
                end_time = time.time()
                duration = end_time - start_time
                start_time = end_time
                print 'epoch:', epoch, 'round:', round, 'train precision@1:', accuracy_,'cost:', cost_, 'duration:', duration
            if round % 1000 == 0:
                self.test_(testset, epoch, round)
            round += 1

    def test_(self, testset, epoch = 0, round = 0):
        #assert(testset.num_features == self.num_features)   
        #num_test_instances = testset.num_instances()
        predicts = []
        cost_ = 0. 
        accuracy_ = 0.
        ground = round
        round = 0
        while True:
          teX, teY = testset.next_batch(batch_size * 10)
          if teX is None:
            break
          feed_dict = melt.gen_feed_dict(self.trainer, self.algo, teX, teY, test_mode = True)
          now_cost, now_accuracy = self.session.run([self.cost, self.accuracy], feed_dict=feed_dict)
          cost_ += now_cost
          accuracy_ += now_accuracy
          round += 1
        cost_ / round
        accuracy_ /= round

        print 'epoch:', epoch, 'round:', ground, 'test  precision@1:', accuracy_,'cost:', cost_ 

        need_stop = False
        #if FLAGS.auto_stop and (accuracy_ - self.avg_accuracy) < FLAGS.min_improve:
        #    need_stop = True 

        self.avg_accuracy = accuracy_

        # if FLAGS.use_summary:
        #     teX, teY = testset.full_batch()
        #     summary_str = self.session.run(merged_summary_op, feed_dict = melt.gen_feed_dict(self.trainer, self.algo, teX, teY, test_mode = True))
        #     summary_writer.add_summary(summary_str, epoch)

        return need_stop

    def save_model(self, model_path, epoch = None):
        self.saver.save(self.session, model_path + '/model.ckpt', global_step = epoch)
    
    def save_info(self, model_path):
        file_ = open(model_path + '/trainer.ckpt', 'w')

        #@FIXME can't pickle module objects error
        #cPickle.dump(self.trainer, file_)
        
        file_.write('%s\t%d\t%d\t%d'%(self.trainer.type, self.trainer.num_features, self.trainer.total_features, self.trainer.index_only))
        
        file_ = open(model_path + '/algo.ckpt', 'w')
        cPickle.dump(self.algo, file_)
        #cPickle.dump(self.algo.options, file_)
        #file_.write('%s'%self.algo.type)

    #since trainer and not dump by cPickle, another way is to load trainset again, but load
    #only one line data, so the save/load will use almost same code as train for reload
    #though a bit more redundant work for reconsructing the graph
    def load(self, model_path):  
        trainer_file = open(model_path + '/trainer.ckpt')
        type_, self.num_features, self.total_features, self.index_only = trainer_file.read().strip().split('\t')
        self.num_features = int(self.num_features)
        self.total_features = int(self.total_features)
        self.index_only = int(self.index_only)
        if type_ == 'dense':
            self.trainer = melt.ClassificationTrainer(num_features=self.num_features, total_features = self.total_features, index_only = self.index_only)
        else:
            self.trainer = melt.SparseClassificationTrainer(num_features=self.num_features) 

        #self.trainer = cPickle.load(trainer_file)
        
        print 'trainer finish loading'        
        
        algo_file = open(model_path + '/algo.ckpt')
        self.algo = cPickle.load(algo_file)  
        #print type(self.algo.options)
        #options = cPickle.load(algo_file)
        #self.algo = self.gen_algo_from_option(options)
        #method = algo_file.read().strip()
        #self.algo = self.gen_algo(method)
        
        print 'algo finish loading ', type(self.algo)
        
        #with tf.device('/cpu:0'):
        self.build_graph(self.algo, self.trainer)
        
        print 'finish building graph'
        
        self.session = tf.Session() 
        
        print 'new session'

        self.saver.restore(self.session, model_path + "/model.ckpt")
        
        print 'dnn predictor finish loading'
        
    def test(self, testset_file):
        testset = melt.load_dataset(testset_file)
        print "finish loading test set ", testset_file
        self.test_(testset)
        

    def predict(self, feature_vecs):
        if type(feature_vecs) != list:
            feature_vecs = [feature_vecs]

        trX = None
        if self.trainer.type == 'sparse':
            trX = melt.sparse_vectors2sparse_features(feature_vecs)
        else: #dense
            trX = melt.dense_vectors2features(feature_vecs, self.trainer.index_only)

        predicts = self.session.run([self.predict_op], feed_dict=melt.gen_feed_dict(self.trainer, self.algo, trX, test_mode = True))
        return predicts
        
    def predict_one(self, feature):
        feature_vecs = [feature]
        score = (self.predict(feature_vecs))[0][0][0]
        return score
    
    def Predict(self, feature, index_only = False):
        #print self.predict_one(libtrate.Vector('24:0.153846,69:0.115385,342:0.666667,354:0.5,409:0.8,420:0.333333,1090:1,1127:0.333333,1241:1,1296:1,1645:0.333333,2058:0.333333,2217:0.333333,6012:0.5,6613:1,6887:1,7350:0.5,9523:0.25,9681:0.25,11030:1,16785:1,21710:0.5,22304:0.5,24282:1,28173:0.25,32809:1,32825:1,51361:1,52573:0.5,52876:1,54153:1,64670:1,95292:1,96030:1,120200:1,213355:1,228161:1,520301:1,797757:1,1191766:1,1263784:1,1263785:1,1263791:1,1263793:1,1263794:1,1263797:1,1263801:1,1263805:1,1263806:1,1263809:1'))
        return self.predict_one(feature)
    
    def Load(self, model_path):
        return self.load(model_path)
    

def predict(classifer, file):
    for line in open(file):
        l = line.strip().split()
        label = l[1]
        feature_str = '\t'.join(l[3:])
        #print label, ' # ' , feature_str
        
        #fe = libtrate.Vector(feature_str)
        
        #print fe.str()
        #print fe.indices.size()
        
        #score = classifer.Predict(fe)
        #print label,' ', score

def main():
    classifer = MultiClassifier()
    if FLAGS.command == 'train':    
        classifer.train(trainset_file, testset_file, method, num_epochs, learning_rate, FLAGS.model)
    elif FLAGS.command == 'test':
        classifer.load(FLAGS.model)
        classifer.test(testset_file)
    elif FLAGS.command == 'predict':
        classifer.load(FLAGS.model)
        predict(classifer, testset_file)
        

if __name__ == "__main__":
    main()

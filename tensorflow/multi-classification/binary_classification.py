#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   binary_classification.py
#        \author   chenghuige  
#          \date   2015-11-30 16:06:52.693026
#   \Description  
# ==============================================================================



import os

# if os.path.abspath('.').startswith('/home/gezi'):
#     import nowarning
#     from libcalibrator import CalibratorFactory 
#     import libtrate
#     import tensorflow as tf 
# else:
import tensorflow as tf 
import nowarning
from libcalibrator import CalibratorFactory 
#import libtrate

import numpy as np

try:
    print 'tf.verion:', tf.__version__
except Exception:
    print 'tf version unknown, might be an old verion'

flags = tf.app.flags
FLAGS = flags.FLAGS

#flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 120, 'Number of epochs to run trainer.')
flags.DEFINE_integer('save_epochs', 50, 'save epochs every save_epochs round')
flags.DEFINE_integer('batch_size', 500, 'Batch size. Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train', './corpus/feature.normed.rand.12000.0_2.txt', 'train file')
flags.DEFINE_string('test', './corpus/feature.normed.rand.12000.1_2.txt', 'test file')
flags.DEFINE_string('method', 'mlp', 'currently support logistic/mlp/mlp2')

flags.DEFINE_boolean('ada_grad', True, 'use ada grad')

flags.DEFINE_boolean('shuffle', False, 'shuffle dataset each epoch')

#----for mlp
flags.DEFINE_integer('hidden_size', 20, 'Hidden unit size')
flags.DEFINE_integer('hidden2_size', 5, 'Hidden unit size of second hidden layer')

flags.DEFINE_string('activation', 'sigmoid', 'you may try tanh or other activate function')
flags.DEFINE_string('activation2', 'sigmoid', 'you may try tanh or other activate function')

#----for cbow
flags.DEFINE_integer('emb_dim', 128, 'embedding dimension')

#first train then test
flags.DEFINE_string('model', './model', 'model path')
flags.DEFINE_string('command', 'train', 'train or test')


#functiona flags
flags.DEFINE_boolean('show_device', False, 'show device info or not')
flags.DEFINE_boolean('use_auc_op', False, 'use user defined auc operator')

flags.DEFINE_boolean('use_summary', False, 'use summary from showing graph')
flags.DEFINE_string('summary_path', '/home/gezi/tmp/tensorflow_logs', 'currently support logistic/mlp/mlp2')

flags.DEFINE_boolean('calibrate', True, 'calibrate the result or not')

flags.DEFINE_boolean('calibrate_trainset', False, 'calibrate by trainset or testset')

flags.DEFINE_boolean('auto_stop', True, 'auto stop iteration and store the best result')
flags.DEFINE_float('min_improve', 0.001 , 'stop when improve less then min_improve')


trainset_file = FLAGS.train
testset_file = FLAGS.test

learning_rate = FLAGS.learning_rate 
num_epochs = FLAGS.num_epochs 
batch_size = FLAGS.batch_size 

method = FLAGS.method


import cPickle

import melt

auc = None

if FLAGS.use_auc_op:
    auc_module = tf.load_op_library('auc.so')  #@TODO @FIXME now has problem of compliling and loading.. 
    #auc = tf.user_ops.auc
    auc = auc_module.auc
else:
    from sklearn.metrics import roc_auc_score 
    auc = roc_auc_score

cout = open('cout.txt', 'w')


from algo import *

class AlgosFactory(object):
    def gen_algo(self, method):
        if method == 'logistic':
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

class BinaryClassifier(object):
    def __init__(self):
        self.calibrator = CalibratorFactory.CreateCalibrator('sigmoid')
        self.auc = 0

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
        #py_x, l2 = algo.forward(trainer)


        Y = trainer.Y
        cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(py_x, Y))
        
        # if hasattr(algo, 'l2'):
        #     cost += 0.0001 * algo.l2
        #cost += 0.1 * l2 

        predict_op = tf.nn.sigmoid(py_x) 
        evaluate_op = None
        if FLAGS.use_auc_op:
            evaluate_op = auc(py_x, Y)

        self.cost = cost
        self.predict_op = predict_op
        self.evaluate_op = evaluate_op  
        #self.weight = algo.weight
        return cost, predict_op, evaluate_op
        
    def foward(self, algo, trainer, learning_rate):
        #with tf.device('/cpu:0'):
        cost, predict_op, evaluate_op = self.build_graph(algo, trainer)

        train_op = None
        if not trainer.index_only:
            if not FLAGS.ada_grad:
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
        
    def calibrate(self, trY, predicts):
        for i in xrange(len(predicts)):
            cout.write('{}\t{}\n'.format(int(trY[i][0] > 0), float(predicts[i][0])))
            self.calibrator.ProcessTrainingExample(float(predicts[i][0]), bool(trY[i][0] > 0), 1.0)

    def train(self, trainset_file, testset_file, method, num_epochs, learning_rate, model_path):
        print 'batch_size:', batch_size, ' learning_rate:', learning_rate, ' num_epochs:', num_epochs
        print 'method:',  method

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

        self.session = tf.Session(config=config) 
        init = tf.initialize_all_variables()
        self.session.run(init)
        

        summary_writer = None
        if FLAGS.use_summary:
            tf.scalar_summary("cross_entropy", self.cost)
            if FLAGS.use_auc_op:
                tf.scalar_summary("auc", evaluate_op)
            merged_summary_op = tf.merge_all_summaries()
            summary_writer = tf.train.SummaryWriter(FLAGS.summary_path, self.session.graph_def)
        
        #os.system('rm -rf ' + FLAGS.model)
        os.system('mkdir -p ' + FLAGS.model)
       
        self.save_info(model_path)

        for epoch in range(num_epochs):
            if epoch > 0 and FLAGS.shuffle:
                trainset = melt.load_dataset(trainset_file)

            self.train_(trainset)
            need_stop = self.test_(testset, epoch = epoch)

            if need_stop:
                print 'need stop as improve is smaller then %f'%FLAGS.min_improve
                break

            #print weight
            #@FIXME 
            if epoch % FLAGS.save_epochs == 0 and not trainer.index_only:
                self.save_model(model_path, epoch)

        self.save_model(model_path)
        if FLAGS.calibrate:
            dataset = trainset
            if not FLAGS.calibrate_trainset:
                dataset = testset
            self.calibrate_(dataset) #@TODO may be test set is right?
            CalibratorFactory.Save(self.calibrator, model_path + '/calibrator.bin')
            #self.calibrator.Save(model_path + '/calibrator.bin')
            self.calibrator.SaveText(model_path + '/calibrator.txt')

        if FLAGS.use_summary:
            teX, teY = testset.full_batch()
            summary_str = self.session.run(merged_summary_op, feed_dict = melt.gen_feed_dict(self.trainer, self.algo, teX, teY, test_mode = True))
            summary_writer.add_summary(summary_str, epoch)
        
        
        #os.system('cp ./{0}/model.ckpt-{1} ./{0}/model.ckpt'.format(FLAGS.model, num_epochs - 1))
    
    def train_(self, trainset):
        num_train_instances = trainset.num_instances()
        #@TODO this minibatch will lost the last instances.. 
        # for start, end in zip(range(0, num_train_instances, batch_size), range(batch_size, num_train_instances, batch_size)):
        #     trX, trY = trainset.mini_batch(start, end)
        #     self.session.run(self.train_op, feed_dict = melt.gen_feed_dict(self.trainer, self.algo, trX, trY))  
        while True:
            trX, trY = trainset.next_batch(batch_size)
            if trX is None:
                break
            self.session.run(self.train_op, feed_dict = melt.gen_feed_dict(self.trainer, self.algo, trX, trY))  

    def test_(self, testset, epoch = 0):
        assert(testset.num_features == self.num_features)   
        auc_ = 0.5
        if self.evaluate_op:
            teX, teY = testset.full_batch()
            predicts, auc_, cost_ = self.session.run([self.predict_op, self.evaluate_op, self.cost], feed_dict = melt.gen_feed_dict(self.trainer, self.algo, teX, teY, test_mode = True))
        else:
            # teX, teY = testset.full_batch()
            # predicts, cost_ = self.session.run([self.predict_op, self.cost], feed_dict = melt.gen_feed_dict(self.trainer, self.algo, teX, teY, test_mode = True))
            num_test_instances = testset.num_instances()
            predicts = []
            cost_ = 0
            while True:
                teX, teY = testset.next_batch(batch_size)
                if teX is None:
                    break
                predicts_, now_cost = self.session.run([self.predict_op, self.cost], feed_dict = melt.gen_feed_dict(self.trainer, self.algo, teX, teY, test_mode = True))
                predicts.extend(predicts_)
                cost_ += now_cost
            teY = testset.labels

            predicts = np.array(predicts)

            #print np.array(zip(teY, predicts))
            #print len(teY), len(predicts)
            auc_ = auc(teY, predicts)

        print epoch, ' auc:', auc_,'cost:', cost_ / len(teY)

        need_stop = False
        if FLAGS.auto_stop and (auc_ - self.auc) < FLAGS.min_improve:
            need_stop = True 

        self.auc = auc_

        return need_stop
            


    def calibrate_(self, dataset):
        num_instances = dataset.num_instances()
        #@TODO better iteration now will lost last group data
        for start, end in zip(range(0, num_instances, batch_size), range(batch_size, num_instances, batch_size)):
            trX, trY = dataset.mini_batch(start, end)
            predicts = self.session.run(self.predict_op, feed_dict = melt.gen_feed_dict(self.trainer, self.algo, trX, trY))  
            self.calibrate(trY, predicts)    
        self.calibrator.FinishTraining()

    def save_model(self, model_path, epoch = None):
        tf.train.Saver().save(self.session, model_path + '/model.ckpt', global_step = epoch)
    
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
        self.calibrator = CalibratorFactory.Load(model_path + '/calibrator.bin')
        trainer_file = open(model_path + '/trainer.ckpt')
        type_, self.num_features, self.total_features, self.index_only = trainer_file.read().strip().split('\t')
        self.num_features = int(self.num_features)
        self.total_features = int(self.total_features)
        self.index_only = int(self.index_only)
        if type_ == 'dense':
            self.trainer = melt.BinaryClassificationTrainer(num_features=self.num_features, total_features = self.total_features, index_only = self.index_only)
        else:
            self.trainer = melt.SparseBinaryClassificationTrainer(num_features=self.num_features) 

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

        tf.train.Saver().restore(self.session, model_path + "/model.ckpt")
        
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
    classifer = BinaryClassifier()
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

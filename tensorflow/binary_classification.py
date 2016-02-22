#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   binary_classification.py
#        \author   chenghuige  
#          \date   2015-11-30 16:06:52.693026
#   \Description  
# ==============================================================================



import os

if os.path.abspath('.').startswith('/home/gezi'):
    import nowarning
    from libcalibrator import CalibratorFactory 
    import libtrate
    import tensorflow as tf 
else:
    import tensorflow as tf 
    import nowarning
    from libcalibrator import CalibratorFactory 
    import libtrate

auc_module = tf.load_op_library('auc.so')
#auc = tf.user_ops.auc
auc = auc_module.auc

try:
    print 'tf.verion:', tf.__version__
except Exception:
    print 'tf version unknown, might be an old verion'
#from sklearn.metrics import roc_auc_score
import cPickle

import melt

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
flags.DEFINE_string('summary_path', '/home/gezi/tmp/tensorflow_logs', 'currently support logistic/mlp/mlp2')

flags.DEFINE_boolean('shuffle', False, 'shuffle dataset each epoch')
flags.DEFINE_boolean('show_device', False, 'show device info or not')
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

trainset_file = FLAGS.train
testset_file = FLAGS.test

learning_rate = FLAGS.learning_rate 
num_epochs = FLAGS.num_epochs 
batch_size = FLAGS.batch_size 

method = FLAGS.method


cout = open('cout.txt', 'w')

print 'batch_size:', batch_size, ' learning_rate:', learning_rate, ' num_epochs:', num_epochs

#from algos import Mlp, MlpOptions, LogisticRegression, LogisticRegressionOptions
from algos import *

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
        else:
            print method, ' is not supported right now, will use default method mlp'
            method = 'mlp'
            return self.gen_algo(method)

class BinaryClassifier(object):
    def __init__(self):
        self.calibrator = CalibratorFactory.CreateCalibrator('sigmoid')
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
        #py_x, self.weight = algo.forward(trainer)
        py_x = algo.forward(trainer)
        Y = trainer.Y
        cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(py_x, Y))
        predict_op = tf.nn.sigmoid(py_x) 
        evaluate_op = auc(py_x, Y)

        self.cost = cost
        self.predict_op = predict_op
        self.evaluate_op = evaluate_op  
        #self.weight = algo.weight
        return cost, predict_op, evaluate_op
        
    def foward(self, algo, trainer, learning_rate):
        #with tf.device('/cpu:0'):
        cost, predict_op, evaluate_op = self.build_graph(algo, trainer)

        if not trainer.index_only:
            train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)   
            #train_op = tf.train.AdagradOptimizer(learning_rate).minimize(cost)    #gpu will fail...
        else:
            ##@TODO WHY  train_op = tf.train.AdagradOptimizer(learning_rate).minimize(cost) will fail...
            #global_step = tf.Variable(0, name="global_step", trainable=False)
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            grads_and_vars = optimizer.compute_gradients(cost)
            train_op = optimizer.apply_gradients(grads_and_vars)
        
        return cost, train_op, predict_op, evaluate_op
        
    def calibrate(self, trY, train_predicts):
        for i in xrange(len(train_predicts)):
            cout.write('{}\t{}\n'.format(int(trY[i][0] > 0), float(train_predicts[i][0])))
            self.calibrator.ProcessTrainingExample(float(train_predicts[i][0]), bool(trY[i][0] > 0), 1.0)

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
        print 'trainer_type:', trainer.type
        print 'trainer_index_only:', trainer.index_only
        
        cost, train_op, predict_op, evaluate_op = self.foward(algo, trainer, learning_rate)
        
        if not FLAGS.show_device:
            self.session = tf.Session() 
        else:
            self.session = tf.Session(config=tf.ConfigProto(log_device_placement=True)) 
        init = tf.initialize_all_variables()
        self.session.run(init)
        
        tf.scalar_summary("cross_entropy", cost)
        tf.scalar_summary("auc", evaluate_op)
        merged_summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(FLAGS.summary_path, self.session.graph_def)
        
        teX, teY = testset.full_batch()
        
        os.system('rm -rf ' + FLAGS.model)
        os.system('mkdir -p ' + FLAGS.model)
        num_train_instances = trainset.num_instances()
        self.save_info(model_path)
        for epoch in range(num_epochs):
            if epoch > 0 and FLAGS.shuffle:
                 trainset = melt.load_dataset(trainset_file)
            for start, end in zip(range(0, num_train_instances, batch_size), range(batch_size, num_train_instances, batch_size)):
                trX, trY = trainset.mini_batch(start, end)
                self.session.run(train_op, feed_dict = melt.gen_feed_dict(self.trainer, self.algo, trX, trY))                 
                #predicts, cost_ = sess.run([predict_op, cost], feed_dict = trainer.gen_feed_dict(teX, teY))
            #print epoch, ' auc:', roc_auc_score(teY, predicts),'cost:', cost_ / len(teY)
    
            predicts, auc_, cost_ = self.session.run([predict_op, evaluate_op, cost], feed_dict = melt.gen_feed_dict(self.trainer, self.algo, teX, teY, test_mode = True))
            #predicts, auc_, cost_, weight = self.session.run([predict_op, evaluate_op, cost, self.weight], feed_dict = trainer.gen_feed_dict(teX, teY))
            print epoch, ' auc:', auc_,'cost:', cost_ / len(teY)
            #print weight
            #@FIXME 
            if epoch % FLAGS.save_epochs == 0 and not trainer.index_only:
                self.save_model(model_path, epoch)
        
        for start, end in zip(range(0, num_train_instances, batch_size), range(batch_size, num_train_instances, batch_size)):
            trX, trY = trainset.mini_batch(start, end)
            train_predicts = self.session.run(predict_op, feed_dict = melt.gen_feed_dict(self.trainer, self.algo, trX, trY))  
            self.calibrate(trY, train_predicts)    
        self.calibrator.FinishTraining()
            
        self.save_model(model_path)
        CalibratorFactory.Save(self.calibrator, model_path + '/calibrator.bin')
        #self.calibrator.Save(model_path + '/calibrator.bin')
        self.calibrator.SaveText(model_path + '/calibrator.txt')

        summary_str = self.session.run(merged_summary_op, feed_dict = melt.gen_feed_dict(self.trainer, self.algo, teX, teY, test_mode = True))
        summary_writer.add_summary(summary_str, epoch)
        
        
        #os.system('cp ./{0}/model.ckpt-{1} ./{0}/model.ckpt'.format(FLAGS.model, num_epochs - 1))
    
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
        assert(testset.num_features == self.num_features)   
        teX, teY = testset.full_batch()
        predicts, auc_, cost_ = self.session.run([self.predict_op, self.evaluate_op, self.cost], feed_dict = melt.gen_feed_dict(self.trainer, self.algo, teX, teY, test_mode = True))
        for i in xrange(len(predicts)):
            print teY[i][0], ' ', predicts[i][0], ' ', self.calibrator.PredictProbability(float(predicts[i][0]))
        print ' auc:', auc_,'cost:', cost_ / len(teY)
        
    def predict(self, feature_vecs):
        if type(feature_vecs) != list:
            feature_vecs = [feature_vecs]
        spf = melt.sparse_vectors2sparse_features(feature_vecs)
        predicts = self.session.run([self.predict_op], feed_dict=melt.gen_feed_dict(self.trainer, self.algo, spf, test_mode = True))
        return predicts
        
    def predict_one(self, feature):
        feature_vecs = [feature]
        return (self.predict(feature_vecs))[0][0][0]
    
    def Predict(self, feature):
        return self.predict_one(feature)
    
    def Load(self, model_path):
        return self.load(model_path)
    

def predict(classifer, file):
    for line in open(file):
        l = line.strip().split()
        label = l[1]
        feature_str = '\t'.join(l[3:])
        #print label, ' # ' , feature_str
        fe = libtrate.Vector(feature_str)
        #print fe.indices.size()
        score = classifer.Predict(fe)
        print label,' ', score

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

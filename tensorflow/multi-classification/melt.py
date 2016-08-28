#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   melt.py
#        \author   chenghuige  
#          \date   2015-11-30 13:40:19.506009
#   \Description  
# ==============================================================================


"""melt is a helper class dealing with both
dense and sparse input(also support sparse and index only format offen used in text classification, like only use word id info no value info),
and try to hide the internal difference of the two input data for user.
It also provides some helper free functions
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

#NUM_CLASSES = 3724
NUM_CLASSES = 34
NUM_FEATURES = 324510

#---------------------------melt load data
def guess_file_format(line):
    """ Gusee file format from the first input line
    Now support melt dense and sparse input file format, for sparse input no header 
    for dense input will ignore header
    @TODO also support libsvm format 
    """
    is_dense = True 
    has_header = False
    index_only = False
    if line.startswith('#'):
        has_header = True
        return  is_dense, has_header, index_only
    elif line.find(':') > 0:
        is_dense = False 
    #no header and dense
    if line.find('.') == -1:
        index_only = True
    return is_dense, has_header, index_only

def guess_label_index(line):
    label_idx = 0
    if line.startswith('_'):
        label_idx = 1
    return label_idx


#@TODO implement [a:b] so we can use [a:b] in application code
class Features(object):
    """Features means DenseFeatures as opposite to SparseFeatures"""
    def __init__(self):
        self.data = []

    def mini_batch(self, start, end):
        return self.data[start: end]

    def full_batch(self):
        return self.data

class SparseFeatures(object):
    def __init__(self):
        self.sp_indices = [] 
        self.start_indices = [0]
        self.sp_ids_val = [] 
        self.sp_weights_val = []
        self.sp_shape = None

    def mini_batch(self, start, end):
        batch = SparseFeatures()
        start_ = self.start_indices[start]
        end_ = self.start_indices[end]
        batch.sp_ids_val = self.sp_ids_val[start_: end_]
        batch.sp_weights_val = self.sp_weights_val[start_: end_]
        row_idx = 0
        max_len = 0
        #@TODO better way to construct sp_indices for each mini batch ?
        for i in xrange(start + 1, end + 1):
            len_ = self.start_indices[i] - self.start_indices[i - 1]
            if len_ > max_len:
                max_len = len_
            for j in xrange(len_):
                batch.sp_indices.append([i - start - 1, j])
            row_idx += 1 
        batch.sp_shape = [end - start, max_len]
        return batch

    def full_batch(self):
        if len(self.sp_indices) == 0:
            row_idx = 0
            max_len = 0
            for i in xrange(1, len(self.start_indices)):
                len_ = self.start_indices[i] - self.start_indices[i - 1]
                if len_ > max_len:
                    max_len = len_
                for j in xrange(len_):
                    self.sp_indices.append([i - 1, j])
                row_idx += 1 
            self.sp_shape = [len(self.start_indices) - 1, max_len]
        return self



class DataSet(object):
    """DataSet handles both dense and sparse features, also support sparse input but with only index 
    """
    def __init__(self):
        self.labels = []
        self.features = None
        self.num_features = 0
        self.total_features = 0
        self.index_only = False
        self.start = 0
        self.num_classes = 2

    def num_instances(self):
        return len(self.labels)

    def full_batch(self):
        return self.features.full_batch(), self.labels

    def mini_batch(self, start, end):
        if end < 0:
            end = self.num_instances() + end 
        return self.features.mini_batch(start, end), self.labels[start: end]

    def next_batch(self, batch_size):
        if self.start == None:
            self.start = 0
            return None, None

        start = self.start
        end = self.start + batch_size
        if end >= self.num_instances():
            end = self.num_instances()
            self.start = None
        else:
            self.start = end

        return self.mini_batch(start, end)


MIN_AFTER_DEQUEUE = 10000
def read(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  return serialized_example

def decode(batch_serialized_examples):
  features = tf.parse_example(
      batch_serialized_examples,
      features={
          'label' : tf.FixedLenFeature([], tf.int64),
          'index' : tf.VarLenFeature(tf.int64),
          'value' : tf.VarLenFeature(tf.float32),
      })

  label = features['label']
  index = features['index']
  value = features['value']

  return label, index, value

def batch_inputs(files, batch_size, num_epochs=None, num_preprocess_threads=1):
  if not num_epochs: num_epochs = None

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
        files, num_epochs=num_epochs)

    serialized_example = read(filename_queue)
    batch_serialized_examples = tf.train.shuffle_batch(
      [serialized_example], 
      batch_size=batch_size, 
      num_threads=num_preprocess_threads,
      capacity=MIN_AFTER_DEQUEUE + 3 * batch_size,
      # Ensures a minimum amount of shuffling of examples.
      min_after_dequeue=MIN_AFTER_DEQUEUE)

    return decode(batch_serialized_examples)
class TfDataSet(object):
    def __init__(self, data_files):
        self.data_files = data_files
        #@TODO now only deal sparse input 
        self.features = SparseFeatures()
        self.label = None

    def build_read_graph(self, batch_size):
        tf_record_pattern = self.data_files
        data_files = tf.gfile.Glob(tf_record_pattern)
        self.label, self.index, self.value = batch_inputs(data_files, batch_size)

    def next_batch(self, sess):
        label, index, value = sess.run([self.label, self.index, self.value])

        trX = (index, value)
        trY = label

        return trX, trY

import random
#dense format the same as melt input inf index_only == False
#also support index only format, may be of differnt length each feature, will be <label, num_features, index0, index1 ...>
#num_features here the same as vocabulary size
#may also be sparse format as <label, num_features, 3:0, 5:0, 3:0, ...>, but here mostly for embedding look up so deal as dense input is fine
def load_dense_dataset(lines, index_only = False):
    random.shuffle(lines)

    dataset_x = []
    dataset_y = []

    nrows = 0
    label_idx = guess_label_index(lines[0])
    total_features = None
    max_label = 0
    for i in xrange(len(lines)):
        if nrows % 10000 == 0:
            print(nrows)
        nrows += 1
        line = lines[i]
        l = line.rstrip().split()
        label = int(l[label_idx])
        if label > max_label:
          max_label = label
        dataset_y.append(label)
        if not index_only:
            dataset_x.append([float(x) for x in l[label_idx + 1:]])
        else:
            dataset_x.append([int(x) for x in l[label_idx + 2:]])
            total_features = int(l[label_idx + 1])
    
    dataset_x = np.array(dataset_x)
    dataset_y = np.array(dataset_y) 

    dataset = DataSet()
    dataset.labels = dataset_y
    #if not index_only:
    #    dataset.num_features = dataset_x.shape[1] 
    #else:
    #    dataset.num_features = num_features
    #    dataset.length = dataset_x.shape[1] 
    dataset.num_features = dataset_x.shape[1]
    if total_features == None:
        total_features = dataset.num_features
    dataset.total_features = total_features
    features = Features()
    features.data = dataset_x
    dataset.features = features
    dataset.index_only = index_only
    dataset.num_classes = max_label + 1
    global NUM_CLASSES
    NUM_CLASSES = max_label + 1
    return dataset

def load_sparse_dataset(lines):
    random.shuffle(lines)
    #print(lines[0])

    dataset_y = []

    label_idx = guess_label_index(lines[0])
    num_features = int(lines[0].split()[label_idx + 1])
    features = SparseFeatures()
    nrows = 0
    start_idx = 0
    max_label = 0;
    for i in xrange(len(lines)):
        if nrows % 10000 == 0:
            print(nrows)
        nrows += 1
        line = lines[i]
        l = line.rstrip().split()
        label = int(l[label_idx])
        #if label > 9:
        #  label = 9
        dataset_y.append(label)
        if label > max_label:
          max_label = label
        start_idx += (len(l) - label_idx - 2)
        features.start_indices.append(start_idx)
        for item in l[label_idx + 2:]:
            id, val = item.split(':')
            features.sp_ids_val.append(int(id))
            features.sp_weights_val.append(float(val))
    dataset_y = np.array(dataset_y)

    dataset = DataSet()
    dataset.labels = dataset_y 
    dataset.num_features = num_features 
    dataset.features = features
    dataset.num_classes = max_label + 1
    global NUM_CLASSES
    NUM_CLASSES = max_label + 1
    print(NUM_CLASSES)
    return dataset


_datasetCache = {}
def load_dataset(dataset, has_header=False, max_lines=0, is_record=False):
    if is_record:
        return TfDataSet(dataset)
    print('... loading dataset:',dataset)
    global _lines
    if dataset not in _datasetCache:
        lines = open(dataset).readlines()
        _datasetCache[dataset] = lines  
    else:
        lines = _datasetCache[dataset]
    if has_header:
     if max_lines <= 0:
         return load_dense_dataset(lines[1:])
     else:
         return load_dense_dataset(lines[1: 1 + max_lines])
    is_dense, has_header, index_only = guess_file_format(lines[0])
    if is_dense:
        if max_lines <= 0:
            return load_dense_dataset(lines[has_header:], index_only)
        else:
            return load_dense_dataset(lines[has_header: has_header + max_lines], index_only)
    else:
        if max_lines <= 0:
            return load_sparse_dataset(lines)
        else:
            return load_sparse_dataset(lines[:max_lines])

#-----------------------------------------melt for tensorflow
import tensorflow as tf

def init_weights(shape, stddev = 0.01, name = None):
    return tf.Variable(tf.random_normal(shape, stddev = stddev), name = name)

def init_bias(shape, val = 0.1, name = None):
  initial = tf.constant(val, shape=shape)
  return tf.Variable(initial, name = name)

def matmul(X, w):
    """ General matmul  that will deal both for dense and sparse input

    Key function for melt.py which hide the differnce of dense adn sparse input for end users
    """
    if type(X) == tf.Tensor:
        return tf.matmul(X,w)
    else:
        return tf.nn.embedding_lookup_sparse(w, X[0], X[1], combiner = "sum")

class ClassificationTrainer(object):
    """General framework for Dense BinaryClassificationTrainer
    """
    def __init__(self, dataset = None, num_features = 0, total_features = 0, index_only = False):
        if dataset is not None:
            self.labels = dataset.labels
            self.features = dataset.features
            self.num_features = dataset.num_features
            self.total_features = dataset.total_features
            #print('length:', self.length)
            self.index_only = dataset.index_only
            self.num_classes = dataset.num_classes
            #print('index_only:', self.index_only)
        else:
            self.num_features = num_features
            self.total_features = total_features
            self.features = Features()
            self.index_only = index_only
            self.num_classes = None
        if not self.index_only:
            self.X = tf.placeholder(tf.float32, [None, self.num_features], name = 'X') 
        else:
            self.X = tf.placeholder(tf.int32, [None, self.num_features], name = 'X') 
        #self.X = tf.placeholder(tf.float32, [None, self.num_features], name = 'X') 
        self.Y = tf.placeholder(tf.int32,  name = 'Y')  #same as batchsize
        
        self.type = 'dense'

    def gen_feed_dict(self, trX, trY=None):
        return {self.X: trX, self.Y: trY}

class SparseClassificationTrainer(object):
    """General framework for Sparse BinaryClassificationTrainer

    Sparse BinaryClassfiction will use sparse embedding look up trick
    see https://github.com/tensorflow/tensorflow/issues/342
    """
    def __init__(self, dataset = None, num_features = 0):
        if dataset is not None and type(dataset) != TfDataSet:
            self.labels = dataset.labels
            self.features = dataset.features
            self.num_features = dataset.num_features
            self.num_classes = dataset.num_classes
        else:
            self.features = SparseFeatures() 
            self.num_features = num_features
            self.num_classes = None

        self.index_only = False
        self.total_features = self.num_features

        if type(dataset) != TfDataSet:
            self.sp_indices = tf.placeholder(tf.int64, name = 'sp_indices')
            self.sp_shape = tf.placeholder(tf.int64, name = 'sp_shape')
            self.sp_ids_val = tf.placeholder(tf.int64, name = 'sp_ids_val')
            self.sp_weights_val = tf.placeholder(tf.float32, name = 'sp_weights_val')
            self.sp_ids = tf.SparseTensor(self.sp_indices, self.sp_ids_val, self.sp_shape)
            self.sp_weights = tf.SparseTensor(self.sp_indices, self.sp_weights_val, self.sp_shape)

            self.X = (self.sp_ids, self.sp_weights)
            self.Y = tf.placeholder(tf.int32) #same as batch size
        else:
            self.X = (dataset.index, dataset.value)
            self.Y = dataset.label
        
        self.type = 'sparse'

    def gen_feed_dict(self, trX, trY=None):
        return {self.Y: trY, self.sp_indices: trX.sp_indices, self.sp_shape: trX.sp_shape,  self.sp_ids_val: trX.sp_ids_val, self.sp_weights_val: trX.sp_weights_val}


def gen_classification_trainer(dataset):
    if type(dataset.features) == Features:
        return ClassificationTrainer(dataset)
    else:
        return SparseClassificationTrainer(dataset)


activation_map = {'sigmoid' :  tf.nn.sigmoid, 'tanh' : tf.nn.tanh, 'relu' : tf.nn.relu}


def gen_feed_dict(trainer, algo, trX, trY=None, test_mode = False):
    if hasattr(algo, 'gen_feed_dict'):
        return algo.gen_feed_dict(trainer, trX, trY, test_mode)
    else:
        return trainer.gen_feed_dict(trX, trY)


#----------------------------for online predict, with input of Vectors
#import libtrate
def sparse_vectors2sparse_features(feature_vecs):
    """This is helper function for converting c++ part melt internal features to SparseFeatures

    will be used for prediction
    """
    spf = SparseFeatures()
    num_features = 0
    max_len = 0
    for feature in feature_vecs:
        len_ = feature.indices.size()
        if len_ > max_len:
            max_len = len_
        if len_ == 0:
            spf.sp_indices.append([num_features, 0])
            spf.sp_ids_val.append(0)
            spf.sp_weights_val.append(0.0)
        else:
            for i in xrange(len_):
                spf.sp_indices.append([num_features, i])
                spf.sp_ids_val.append(int(feature.indices[i]))
                spf.sp_weights_val.append(float(feature.values[i]))
        num_features += 1
    spf.sp_shape = [num_features, max_len]
    #print('spf.sp_shape:', spf.sp_shape)
    #print('spf.sp_ids_val', spf.sp_ids_val)
    return spf

def dense_vectors2features(feature_vecs, index_only):
    """This is helper function for converting c++ part melt internal features to DenseFeatures

    will be used for prediction
    """
    dataset_x = []

    for feature in feature_vecs:
        if not index_only:
            dataset_x.append([float(x) for x in feature.values])
        else:
            dataset_x.append([int(x) for x in feature.values])

    return np.array(dataset_x)

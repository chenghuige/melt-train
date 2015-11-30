#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   melt.py
#        \author   chenghuige  
#          \date   2015-11-30 13:40:19.506009
#   \Description  
# ==============================================================================

import numpy as np
import os 

#---------------------------melt load data
#Now support melt dense and sparse input file format, for sparse input no
#header
#for dense input will ignore header
#also support libsvm format @TODO
def guess_file_format(line):
	is_dense = True 
	has_header = False
	if line.startswith('#'):
		has_header = True
		return  is_dense, has_header
	elif line.find(':') > 0:
		is_dense = False 
	return is_dense, has_header

def guess_label_index(line):
	label_idx = 0
	if line.startswith('_'):
		label_idx = 1
	return label_idx


#@TODO implement [a:b] so we can use [a:b] in application code
class Features(object):
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
	def __init__(self):
		self.labels = []
		self.features = None
		self.num_features = 0

	def num_instances(self):
		return len(self.labels)

	def full_batch(self):
		return self.features.full_batch(), self.labels

	def mini_batch(self, start, end):
		if end < 0:
			end = num_instances() + end 
		return self.features.mini_batch(start, end), self.labels[start: end]

def load_dense_dataset(lines):
	dataset_x = []
	dataset_y = []

	nrows = 0
	label_idx = guess_label_index(lines[0])
	for i in xrange(len(lines)):
		if nrows % 10000 == 0:
			print nrows
		nrows += 1
		line = lines[i]
		l = line.rstrip().split()
		dataset_y.append([float(l[label_idx])])
		dataset_x.append([float(x) for x in l[label_idx + 1:]])
	
	dataset_x = np.array(dataset_x)
	dataset_y = np.array(dataset_y) 

	dataset = DataSet()
	dataset.labels = dataset_y
	dataset.num_features = dataset_x.shape[1]
	features = Features()
	features.data = dataset_x
	dataset.features = features
	return dataset

def load_sparse_dataset(lines):
	dataset_x = []
	dataset_y = []

	label_idx = guess_label_index(lines[0])
	num_features = int(lines[0].split()[label_idx + 1])
	features = SparseFeatures()
	nrows = 0
	start_idx = 0
	for i in xrange(len(lines)):
		if nrows % 10000 == 0:
			print nrows
		nrows += 1
		line = lines[i]
		l = line.rstrip().split()
		dataset_y.append([float(l[label_idx])])
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
	return dataset

def load_dataset(dataset, has_header=False):
	print '... loading dataset:',dataset
	lines = open(dataset).readlines()
	if has_header:
		return load_dense_dataset(lines[1:])
	is_dense, has_header = guess_file_format(lines[0])
	if is_dense:
		return load_dense_dataset(lines[has_header:])
	else:
		return load_sparse_dataset(lines)

#-----------------------------------------melt for tensorflow
import tensorflow as tf

def init_weights(shape):
	return tf.Variable(tf.random_normal(shape, stddev = 0.01))

def matmul(X, w):
	if type(X) == tf.Tensor:
		return tf.matmul(X,w)
	else:
		return tf.nn.embedding_lookup_sparse(w, X[0], X[1], combiner = "sum")

class BinaryClassificationTrainer(object):
	def __init__(self, dataset):
		self.labels = dataset.labels
		self.features = dataset.features
		self.num_features = dataset.num_features

		self.X = tf.placeholder("float", [None, self.num_features]) 
		self.Y = tf.placeholder("float", [None, 1])

	def gen_feed_dict(self, trX, trY):
		return {self.X: trX, self.Y: trY}

class SparseBinaryClassificationTrainer(object):
	def __init__(self, dataset):
		self.labels = dataset.labels
		self.features = dataset.features
		self.num_features = dataset.num_features

		self.sp_indices = tf.placeholder(tf.int64)
		self.sp_shape = tf.placeholder(tf.int64)
		self.sp_ids_val = tf.placeholder(tf.int64)
		self.sp_weights_val = tf.placeholder(tf.float32)
		self.sp_ids = tf.SparseTensor(self.sp_indices, self.sp_ids_val, self.sp_shape)
		self.sp_weights = tf.SparseTensor(self.sp_indices, self.sp_weights_val, self.sp_shape)

		self.X = (self.sp_ids, self.sp_weights)
		self.Y = tf.placeholder("float", [None, 1])

	def gen_feed_dict(self, trX, trY):
		return {self.Y: trY, self.sp_indices: trX.sp_indices, self.sp_shape: trX.sp_shape,  self.sp_ids_val: trX.sp_ids_val, self.sp_weights_val: trX.sp_weights_val}


def gen_binary_classification_trainer(dataset):
	if type(dataset.features) == Features:
		return BinaryClassificationTrainer(dataset)
	else:
		return SparseBinaryClassificationTrainer(dataset)

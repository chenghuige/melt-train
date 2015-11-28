#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   melt_dataset.py
#        \author   chenghuige  
#          \date   2015-11-17 10:55:40.916390
#   \Description  
# ==============================================================================

import numpy as np
import os 

def parse_first_line(line):
	label_idx = 0
	if line.startswith('_'):
		label_idx = 1
	return label_idx

def load_dense_data(dataset, has_header = False):
	''' Loads the dataset

	:type dataset: string
	:param dataset: the path to the dataset 
	'''
	print '... loading data:',dataset
	dataset_x = []
	dataset_y = []

	lines = open(dataset).readlines()
	if (lines[0].startswith('#')):
		has_header = True
	
	label_idx = parse_first_line(lines[has_header]) 
	nrows = 0
	for i in xrange(has_header, len(lines)):
		if nrows % 10000 == 0:
			print nrows
		nrows += 1
		line = lines[i]
		l = line.rstrip().split()
		dataset_y.append([float(l[label_idx])])
		dataset_x.append([float(x) for x in l[label_idx + 1:]])
	
	dataset_x = np.array(dataset_x)
	dataset_y = np.array(dataset_y) 
	return dataset_x, dataset_y

def load_sparse_data(dataset):
	print '... loading data:',dataset
	dataset_x = []
	dataset_y = []

	lines = open(dataset).readlines()
	label_idx = parse_first_line(lines[0])
	num_features = int(lines[0].split()[label_idx + 1])
	nrows = 0
	for i in xrange(len(lines)):
		if nrows % 10000 == 0:
			print nrows
		nrows += 1
		line = lines[i]
		l = line.rstrip().split()
		dataset_y.append([float(l[label_idx])])
		dataset_x.append(l[label_idx + 2:])
	dataset_y = np.array(dataset_y)
	return dataset_x, dataset_y, num_features

def sparse2dense(dataset_x, num_features):
	print "start convert to dense"
	dataset_x_ = []
	for instance in dataset_x:
		l = [float(0)] * num_features
		for item in instance:
			index, value = item.split(':')
			l[int(index)] = float(value)
		dataset_x_.append(l)
	print "finish convert to dense"
	return np.array(dataset_x_)


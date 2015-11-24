#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   melt_dataset.py
#        \author   chenghuige  
#          \date   2015-11-17 10:55:40.916390
#   \Description  
# ==============================================================================

import numpy as np
import cPickle
import os 

def load_dense_data(dataset, has_header = False):
	''' Loads the dataset

	:type dataset: string
	:param dataset: the path to the dataset 
	'''
	print '... loading data:',dataset
	dataset_x = []
	dataset_y = []
	#cache_file = ''
	#if dataset.endswith('.txt'):
	#	cache_file = dataset.replace('.txt', '.pkl')
	#else:
	#	cache_file = dataset + '.pkl'
	#if os.path.isfile(cache_file):
	#	print 'loading from cache file directly'
	#	dataset_x, dataset_y= cPickle.load(open(cache_file, 'rb'))
	#	return dataset_x, dataset_y

	lines = open(dataset).readlines()
	print 'load all lines to memory'
	if (lines[0].startswith('#')):
		has_header = True
	nrows = 0
	for i in xrange(has_header, len(lines)):
		if nrows % 10000 == 0:
			print nrows
		nrows += 1
		line = lines[i]
		l = line.rstrip().split()
		label_idx = 0
		if l[0].startswith('_'):
			label_idx = 1
		dataset_y.append([float(l[label_idx])])
		dataset_x.append([float(x) for x in l[label_idx + 1:]])
	
	dataset_x = np.array(dataset_x)
	dataset_y = np.array(dataset_y) 
	#cPickle.dump((dataset_x, dataset_y), open(cache_file, 'wb'))
	return dataset_x, dataset_y


 


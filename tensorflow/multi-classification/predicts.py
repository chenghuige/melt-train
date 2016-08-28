#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   predict.py
#        \author   chenghuige  
#          \date   2016-03-12 10:29:35.146422
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys,os

import nowarning
#@FIXME must before libtieba..
from predictors.mlp_predictor import TextPredictor  
import libtieba

from libtieba import LogHelper

LogHelper.set_level(4)


model_path = './models/logistic'

if len(sys.argv) > 1:
	model_path = sys.argv[1]

print('model_path:%s'%model_path)

predictor = TextPredictor(model_path)

for line in open('./posts.txt').readlines():
	l = line.split('\t')
	pid = l[0]
	title = l[1]
	content = l[2]
	print(pid)
	print(title)
	print(content)

	adjusted_score, score = predictor.predict(title, content)

	print('adjusted_score:%f\tscore:%f\n'%(adjusted_score, score))
  

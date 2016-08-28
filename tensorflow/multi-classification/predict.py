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


predictor = TextPredictor('./model')

pid = 85358381008
title = ''
content = ''

#info = libtieba.get_post_info(pid)
#title = info.title
#content = info.content 

if title == '' and content == '':
	line = open('./posts.txt').readlines()[-1]
	l = line.split('\t')
	pid = l[0]
	title = l[1]
	content = l[2]

title = '一个人，一台单反，做着一份很轻松的工作，一个人的旅行'
content = ''
print(pid)
print(title)
print(content)


adjusted_score, score = predictor.predict(title, content)

print('adjusted_score:%f\tscore:%f\n'%(adjusted_score, score))
  

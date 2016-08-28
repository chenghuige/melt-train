#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   cnn_util.py
#        \author   chenghuige  
#          \date   2016-03-18 11:54:49.160954
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import nowarning 

from libgezi import *
from libsegment import *
#from libidf import *
from libtieba import *

from trate_conf import *

LogHelper.set_level(4)
Segmentor.Init()

max_title_len = 72
max_content_len = 102

tpad = '<tpad/>'
cpad = '<cpad/>'

tcpad = '<tcpad/>'
cepad = '<cepad/>'

def title2cnnvec(title):
	global max_title_len, tpad, tcpad
	title = get_real_title(title)
	tvec = Segmentor.Segment(title, seg_type)
	tvec.push_back(tpad)
	tvec2 = Segmentor.Segment(normalize_str(title), seg_type)
	for word in tvec2:
		tvec.push_back(word)
	tvec.resize(max_title_len, tcpad)
	return tvec

def content2cnnvec(content):
	global max_len, max_content_len, tcpad, cepad
	content = strip_html(content)
	if len(content) > max_len:
		content = gbk_substr(content, 0, max_len)
	cvec = Segmentor.Segment(content, seg_type)
	cvec.push_back(cpad)
	cvec2 = Segmentor.Segment(normalize_str(content), seg_type)
	for word in cvec2:
		cvec.push_back(word)
	cvec.resize(max_content_len, cepad)
	return cvec
  

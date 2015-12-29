#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   load.py
#        \author   chenghuige  
#          \date   2015-12-29 10:51:04.414808
#   \Description  
# ==============================================================================

import sys,os

import test_pickle

import cPickle

x = cPickle.load(open('./a.txt'))
print x.a 
print type(x)
 

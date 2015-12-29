#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   test2.py
#        \author   chenghuige  
#          \date   2015-12-29 10:13:56.161794
#   \Description  
# ==============================================================================

import sys,os

import test_pickle

import cPickle

#test_pickle.save()
x = cPickle.load(open('./a.txt'))

print x.a
print type(x)

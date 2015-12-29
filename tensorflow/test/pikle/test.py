#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   test.py
#        \author   chenghuige  
#          \date   2015-12-29 10:13:48.212088
#   \Description  
# ==============================================================================

import sys,os

class A(object):
	def __init__(self):
		self.a = 3

import cPickle

x = A()
cPickle.dump(x, open('a.txt', 'w'))

y = cPickle.load(open('a.txt'))

print y.a

def save():
	x = A()
	cPickle.dump(x, open('b.txt', 'w'))

save()
z = cPickle.load(open('b.txt'))

print z.a 


#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   deal.py
#        \author   chenghuige  
#          \date   2015-03-25 11:46:36.285629
#   \Description  
# ==============================================================================

import sys,os

for line in open(sys.argv[1]):
	print line.replace(',', '\t'),

 

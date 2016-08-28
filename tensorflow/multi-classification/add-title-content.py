#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   add-title-content.py
#        \author   chenghuige  
#          \date   2016-03-14 10:12:16.797568
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys,os

import nowarning 
import libtieba 

pid = int(sys.argv[1])
info = libtieba.get_post_info(pid)

title = info.title.replace('\n', ' ')
content = info.content.replace('\n', ' ')

with open('posts.txt', 'a') as f:
	f.write('%d\t%s\t%s\n'%(pid, title, content))
  

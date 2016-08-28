#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   trate-conf.py
#        \author   chenghuige  
#          \date   2015-06-29 17:46:36.253018
#   \Description  
# ==============================================================================

import nowarning
import libsegment

sep = '\x01'
seg_type = libsegment.SEG_BASIC
#seg_type = libsegment.SEG_WPCOMP
ngram = 1
skip = 0
#max_len = 1024 
max_len = 100

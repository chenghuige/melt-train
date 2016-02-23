#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   gen-so-all.py
#        \author   chenghuige  
#          \date   2016-02-17 11:30:20.329396
#   \Description  
# ==============================================================================

import sys,os

import glob

for _file in glob.glob('*.cc'):
    command = 'sh ./gen-so.sh ./{}'.format(_file)
    print command 
    os.system(command)

 

#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   print.py
#        \author   chenghuige  
#          \date   2015-03-20 19:25:24.960651
#   \Description  
# ==============================================================================

import sys,os

from prettytable import PrettyTable

x = PrettyTable(['#id', 'label', 'hight', 'money', 'face'])
x.add_row(['_0', 1, 20, 80, 100])
x.add_row(['_1', 1, 60, 90, 25])
x.add_row(['_2', 1, 3, 95, 95])
x.add_row(['_3', 1, 66, 95, 60])
x.add_row(['_4', 0, 30, 95, 25])
x.add_row(['_5', 0, 20, 12, 55])
x.add_row(['_6', 0, 15, 14, 99])
x.add_row(['_7', 0, 10, 99, 2])

print x

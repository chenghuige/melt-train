
# coding: utf-8

# In[1]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys,os

import nowarning
#@FIXME must before libtieba..
from predictors.cnn_predictor import TextPredictor  
import libtieba

predictor = TextPredictor('./model')

import gezi
#gezi.ENCODE = 'utf8'
from gezi import gprint
from gezi import togbk
from gezi import toutf8



title = togbk('')
content = togbk('')
adjusted_score, score = predictor.predict(title, content)
print('adjusted_score:%f\tscore:%f\n'%(adjusted_score, score))

title = '��һУһԺһʦһ��һ��һ��һ'
content = '��һУһԺһʦһ��һ��һ��һ��һQһQһȺ�ţ�5234622 ��һ��һ��һʦһ��һ��һ��һ��һ��һ��һ��һ��һ��һ��һ��һȺһ��һ��һ��һ��һǪһ��һ����'
adjusted_score, score = predictor.predict(title, content)
print('adjusted_score:%f\tscore:%f\n'%(adjusted_score, score))



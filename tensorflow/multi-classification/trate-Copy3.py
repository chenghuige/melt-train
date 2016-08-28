
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

title = '高一校一院一师一姐一姓一生一'
content = '高一校一院一师一姐一姓一生一活一Q一Q一群号：5234622 有一四一百一师一姐一妹一开一放一的一身一体一等一待一安一抚一群一内一还一有一上一仟一文一件。'
adjusted_score, score = predictor.predict(title, content)
print('adjusted_score:%f\tscore:%f\n'%(adjusted_score, score))



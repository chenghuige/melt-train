#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   bc_predict.py
#        \author   chenghuige  
#          \date   2015-12-29 13:51:02.222796
#   \Description  
# ==============================================================================



from binary_classification import BinaryClassifier

import nowarning
import libtieba, libnormalizer, libcalibrator

bc = BinaryClassifier()

bc.load('./model')
 
pid = 81431360452
pid = 81431361392
pid = 81412038123

info = libtieba.get_post_info(pid)

print info.title, ' ', info.content 
 
import libtrate


identifer = libtrate.DoubleIdentifer()
identifer.Load('./data/ltrate.thread.model/identifer.bin')
print identifer.size()

#normalizer = libtrate.NormalizerFactory.CreateNormalizer('minmax', './data/ltrate.thread.model/normalizer.bin')
normalizer = libnormalizer.NormalizerFactory.Load('./data/ltrate.thread.model/normalizer.bin')
lpredictor = libtrate.PredictorFactory.LoadPredictor('./data/ltrate.thread.model/')

from libcalibrator import CalibratorFactory
calibrator = CalibratorFactory.Load('./model/calibrator.bin')

print type(normalizer)

import libgezi
def deal_content(content):
    content = libgezi.strip_html(content)
    if len(content) > 100:
        content = libgezi.gbk_substr(content, 0, 100)
        content = content + ' ' + libgezi.normalize_str(content)
    return content

def deal_title(title):
    return title + ' ' + libgezi.normalize_str(title)
    
title = deal_title(info.title)
print title

content = deal_content(info.content)
print content

from libsegment import *
from libsegment import LogHelper

LogHelper.set_level(4)

Segmentor.Init()
title_words = Segmentor.Segment(title, SEG_BASIC)
content_words = Segmentor.Segment(content, SEG_BASIC)
id_val_map =  libtrate.id_map()
num_words = identifer.size()
libtrate.TextPredictor.Prase(title_words, id_val_map, identifer, 0)
libtrate.TextPredictor.Prase(content_words, id_val_map, identifer, num_words)
print id_val_map.size()
fe = libtrate.Vector(id_val_map)
print fe.indices.size() 
print fe.str()
#@FIXME why wrong core.....??
fe = normalizer.NormalizeCopy(fe)
#fe = lpredictor.GetNormalizer().NormalizeCopy(fe)
print fe.str() 

score = float(bc.Predict(fe))
print 'score:{}, adjusted_score:{}'.format(score, calibrator.PredictProbability(score))
print 'score2:', libtrate.TextPredictor.Predict(title_words, content_words, identifer, lpredictor)

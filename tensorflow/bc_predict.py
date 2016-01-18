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



 
pid = 81431360452
pid = 81431361392
pid = 81412038123
pid = 81455458692
pid = 81437135034
#pid = 81605089091
#pid = 81605080188
pid = 81720185529

#pid = 68942560362

#pid = 81883993132
#pid = 81888138226
#pid = 81953081444

#pid = 81953350152
#pid = 81986525508
pid = 82069911145 

#pid = 82069969465

pid = 82069968445

pid = 82073369485

pid = 82154598133

info = libtieba.get_post_info(pid)

print info.title, ' ', info.content 
 
import libtrate


identifer = libtrate.DoubleIdentifer()
#identifer.Load('./data/ltrate.thread.model/identifer.bin')
identifer.Load('./ltrate.thread.model/identifer.bin')
print identifer.size()
print identifer.id('工程')

#normalizer = libtrate.NormalizerFactory.CreateNormalizer('minmax', './data/ltrate.thread.model/normalizer.bin')
#normalizer = libnormalizer.NormalizerFactory.Load('./data/ltrate.thread.model/normalizer.bin')
normalizer = libnormalizer.NormalizerFactory.Load('./ltrate.thread.model/normalizer.bin')
#lpredictor = libtrate.PredictorFactory.LoadPredictor('./data/ltrate.thread.model/')
lpredictor = libtrate.PredictorFactory.LoadPredictor('./ltrate.thread.model/')

from libcalibrator import CalibratorFactory
calibrator = CalibratorFactory.Load('./model/calibrator.bin')


bc = BinaryClassifier()

bc.load('./model')

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

#info.title = '发一贴，btv2，佳一，商毅，高雷雷及央视张斌。'
#info.title = 'btv2，佳一，商毅，高雷雷及央视张斌。'
#info.title = '佳一，商毅，高雷雷及央视张斌。'
#info.title = '商毅，高雷雷及央视张斌。'
#info.title = '高雷雷及央视张斌。'
#info.title = '央视'
#info.title = '张斌。'
#info.title = '央视张斌。'
#info.title = '斌。'
#info.title = '。'
title = deal_title(info.title)
print title

#info.content = ''
content = deal_content(info.content)
print content

from libsegment import *
from libsegment import LogHelper

LogHelper.set_level(4)

Segmentor.Init()
title_words = Segmentor.Segment(title, SEG_BASIC)
print '\x01'.join(title_words)
content_words = Segmentor.Segment(content, SEG_BASIC)
print '\x01'.join(content_words)
id_val_map =  libtrate.id_map()
num_words = identifer.size()

#for i in range(title_words.size()):
#	if title_words[i] == '害人':
#		title_words[i] = ' '
#title_words.clear()
#content_words.clear()
#title_words.push_back('害人')
#content_words.push_back('害人')
libtrate.TextPredictor.Prase(title_words, id_val_map, identifer, 0, ngram = 3, skip = 2)
libtrate.TextPredictor.Prase(content_words, id_val_map, identifer, num_words, ngram = 3, skip = 2)
#libtrate.TextPredictor.Prase(title_words, id_val_map, identifer, 0)
#libtrate.TextPredictor.Prase(content_words, id_val_map, identifer, num_words)
print id_val_map.size()
fe = libtrate.Vector(id_val_map)
print fe.str()
#fe2 = fe
#print 'score3:', lpredictor.Predict(fe2)
#print 'fe2: ', fe2.str()
#print fe.str()
#@FIXME why wrong core.....??
#fe = normalizer.NormalizeCopy(fe)
fe = lpredictor.GetNormalizer().NormalizeCopy(fe)
print fe.str() 

score = float(bc.Predict(fe))
print score
print 'score:{}, adjusted_score:{}'.format(score, calibrator.PredictProbability(score))
print 'score2:', libtrate.TextPredictor.Predict(title_words, content_words, identifer, lpredictor)

indexes = []
for index in fe.indices:
	indexes.append(str(index))

indexes_str = ' '.join(indexes)

lego_line = '{};{};{}\n'.format(pid, indexes_str, 0)
out = open('lego.txt', 'w')
out.write(lego_line)


#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   trate_predictor.py
#        \author   chenghuige  
#          \date   2015-12-18 11:42:52.080763
#   \Description  
# ==============================================================================

import sys,os
import collections

#@TODO this is a bug as lib... which is generated using boost.python should import after tensorflow
#if not will core during seesion restore
#from binary_classification import BinaryClassifier as Predictor

import nowarning
import libtrate, libnormalizer
import libgezi 
from libsegment import *

#from binary_classification import MlpOptions

from libmelt_predict import PredictorFactory

class TextPredictor(object):
    def __init__(self, model_path):
        self.identifer = libtrate.DoubleIdentifer()
        self.identifer.Load(model_path + '/identifer.bin')

        #self.predictor = Predictor()
        #self.predictor.load(model_path)
        self.predictor = PredictorFactory.LoadPredictor('./lego.model/')
        self.normalizer = libnormalizer.NormalizerFactory().Load(model_path + '/normalizer.bin')
     
        #from libcalibrator import CalibratorFactory
        #self.calibrator = CalibratorFactory.Load(model_path + '/calibrator.bin')

        Segmentor.Init()


    #def deal_title(self, title):
    #    return title + ' ' + libgezi.normalize_str(title)
    def deal_title(self, title):
        return libgezi.normalize_str(title)

    #def deal_content(self, content):
    #    content = libgezi.strip_html(content)
    #    if len(content) > 100:
    #        content = libgezi.gbk_substr(content, 0, 100)
    #    content = content + ' ' + libgezi.normalize_str(content)
    #    return content
    def deal_content(self, content):
        content = libgezi.strip_html(content)
        if len(content) > 100:
            content = libgezi.gbk_substr(content, 0, 100)
        content = libgezi.normalize_str(content)
        return content

    def predict(self, title, content):
        Segmentor.Init()
        title = self.deal_title(title)
        content = self.deal_content(content)
        title_words = Segmentor.Segment(title, SEG_BASIC)
        content_words = Segmentor.Segment(content, SEG_BASIC)

        id_val_map =  libtrate.id_map()
        num_words = self.identifer.size()
        libtrate.TextPredictor.Prase(title_words, id_val_map, self.identifer, 0, ngram = 1, skip = 0)
        libtrate.TextPredictor.Prase(content_words, id_val_map, self.identifer, num_words, ngram = 1, skip = 0)


        fe = libtrate.Vector(id_val_map)
        normed_fe = self.normalizer.NormalizeCopy(fe)

        score = float(self.predictor.Predict(normed_fe))
        #adjusted_score = self.calibrator.PredictProbability(score)
  
        #return adjusted_score     
        score = 1 - score
        return score 

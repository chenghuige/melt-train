#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   run.py
#        \author   chenghuige  
#          \date   2015-12-17 16:54:03.137910
#   \Description  
# ==============================================================================

import sys,os

import gflags
FLAGS = gflags.FLAGS

gflags.DEFINE_string('conf_file', 'redis_client.conf', '')
gflags.DEFINE_string('conf_dir', './conf', '')
gflags.DEFINE_string('thread_key', '#!thread!#', '')
gflags.DEFINE_integer('queue_size', 100, 'fetch num each time from redis')
gflags.DEFINE_integer('set_capacity', 10000000, 'fetch num each time from redis')
gflags.DEFINE_float('thre', 0.5, 'fetch num each time from redis')
gflags.DEFINE_string('model', './model', 'model path')

FLAGS(sys.argv)

from mlp_predictor import TextPredictor

import nowarning
import libredis 
libredis.LogHelper.set_level(4)
import libtieba

pid_set = set()
redis_client = libredis.RedisClient()
ret = redis_client.Init(FLAGS.conf_file, FLAGS.conf_dir)

if ret != 0:
    print 'redis init fail ret:', ret
    exit(-1)

predictor = TextPredictor(FLAGS.model)

num_normals = 0
num_spams = 0

def run(pids):
    global predictor, num_normals, num_spams
    infos = libtieba.get_posts_info(pids)
    for info in infos:
        score = predictor.predict(info.title, info.content)
        if score > FLAGS.thre:
            num_spams += 1
            print info.postId, info.threadId, info.forumName, info.userName, info.title, info.content
            print 'score: ', score, ' spam ratio: ', float(num_spams) / (num_spams + num_normals), FLAGS.thre
        else:
            num_normals += 1
        l = [str(score), str(info.postId), info.forumName, info.userName, info.title, info.content]
        line = '\t'.join(l)
         
round = 0
while True:
    vec = libredis.svec()
    deal_pids = libredis.ulvec()
    ret = redis_client.ZrangeFirstNElement(FLAGS.thread_key, FLAGS.queue_size, vec)
    if ret != 0:
        print 'read redis fail ret:', ret
    #print len(vec)
    num_old = 0
    num_new = 0 
    for pid in vec:
        if pid in pid_set:
            num_old += 1
        else:
            num_new += 1
            pid_set.add(pid)
            if len(pid_set) > FLAGS.set_capacity:
                pid_set.clear()
            deal_pids.append(int(pid))
    #print 'new:', num_new, ' old:', num_old, ' total:', len(pid_set)
    run(deal_pids)
    round += 1
 

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
gflags.DEFINE_float('thre', 0.98, 'fetch num each time from redis')
gflags.DEFINE_string('model', './model', 'model path')
gflags.DEFINE_string('command_no', '30002', 'write to which db section')
gflags.DEFINE_integer('buffer_size', 10, 'write db buffer size')

gflags.DEFINE_boolean('write_db', True, 'write info to db')
gflags.DEFINE_boolean('realtime_del', True, 'use realtime del or not')

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

import gezi
import libtieba
import MySQLdb,urllib,json
class DbWriter(object):
    def __init__(self, ip = '10.99.89.42', command_no = '30002', buffer_size = 100):
        self.conn = MySQLdb.connect(host='10.99.89.42',user='root',passwd='root',db='tieba',port=3316)
        self.cur = self.conn.cursor()
        self.cur.execute('set names gbk')
        self.sql = "insert ignore into dm_monitor(forum_id, thread_id, post_id, monitor_type, create_time, title, content, user, forum, user_id, ip) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        self.vals = []
        self.tids = libtieba.ulvec()
        self.buffer_size = buffer_size
        self.command_no = command_no

    def add(self, info, score):
        create_time_ = gezi.get_timestr(info.createTime)
        self.tids.push_back(info.threadId)
        content = '[score:{} thre:{}]{}'.format(score, FLAGS.thre, info.content)
        val = [str(info.forumId), str(info.threadId), str(info.postId) , self.command_no, create_time_, info.title, content, info.userName, info.forumName, str(info.userId), str(info.ip)]
        self.vals.append(val)

        vals_ = self.vals 

        if len(self.vals) == self.buffer_size:
            self.write(self.vals)
            self.vals = []
            self.tids = libtieba.ulvec()

        return vals_

    def write(self, vals):
        print 'writting to db count: ' + str(len(vals))

        deletedTids = libtieba.is_threads_deleted(self.tids)
        content_idx = 6
        for i in xrange(self.tids.size()):
            if self.tids[i] in deletedTids:
                vals[i][content_idx] = 'idDeleted:1 ' + vals[i][content_idx] 
            else:
                vals[i][content_idx] = 'isDeleted:0 ' + vals[i][content_idx]

        try:
            self.cur.executemany(self.sql, tuple(vals))
        except Exception, e:
            print e

        self.conn.commit()
        #print 'finish writting to db'

    def close(self):
        self.cur.close()
        self.conn.close()

    def vals(self):
        return self.vals

db_writer = DbWriter(command_no = FLAGS.command_no, buffer_size = FLAGS.buffer_size)

def run(pids):
    #print 'run pids begin'
    global predictor, num_normals, num_spams
    infos = libtieba.get_posts_info(pids)
    #print 'finish get posts info'
    for info in infos:
        score = predictor.predict(info.title, info.content)
        #print 'finish predict'
        if score > FLAGS.thre:
            num_spams += 1
            print info.postId, info.threadId, info.forumName, info.userName, info.title, info.content
            print 'score: ', score, ' spam ratio: ', float(num_spams) / (num_spams + num_normals), FLAGS.thre
            vals_ = []
            if FLAGS.write_db:
                vals_ = db_writer.add(info, score)
                #print 'finish add one to db'
            if FLAGS.realtime_del:
                if len(vals_) == FLAGS.buffer_size:
                    print 'real time delete'
                    exe = '/home/users/chenghuige/forum/orp002/php/bin/php'
                    out = open('delete.temp', 'w')
                    for val in vals_:
                        thread_id = val[1]
                        post_id = val[2]
                        user_id = val[-2]
                        forum_id = val[0]
                        out.write('{}\t{}\t{}\t{}\n'.format(thread_id, post_id, user_id, forum_id))
                    out.close()
                    os.system(exe + ' multi-delete.php delete.temp')
        else:
            num_normals += 1
        l = [str(score), str(info.postId), info.forumName, info.userName, info.title, info.content]
        line = '\t'.join(l)
    #print 'run pids end'
         
round = 0
while True:
    vec = libredis.svec()
    deal_pids = libredis.ulvec()
    ret = redis_client.ZrangeFirstNElement(FLAGS.thread_key, FLAGS.queue_size, vec)
    if ret != 0:
        print 'read redis fail ret:', ret
    #print len(vec)
    #print 'finish redis read ok'
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
 

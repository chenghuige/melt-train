#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   write-db.py
#        \author   chenghuige  
#          \date   2014-07-01 13:58:28.897098
#   \Description  
# ==============================================================================

import sys,os
import gezi #must before MySQLdb why...
import MySQLdb,urllib,json
 
command_no = '90023'

conn = MySQLdb.connect(host='10.99.89.42',user='root',passwd='root',db='tieba',port=3316)
cur = conn.cursor()
cur.execute('set names gbk')

sql = "insert ignore into dm_monitor(forum_id, thread_id, post_id, monitor_type, create_time, title, content, user, forum, user_id, ip) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"

vals=[]
for line in open(sys.argv[1]):
	count = 1
	print line,
	l = line.strip().split('\t')
	thread_id, post_id, uid, forum_id, ip, create_time, score, title, content, forum, uname, now_time, num_posts, thre = line.strip().split('\t')
	create_time_ = gezi.get_timestr(int(create_time))
	now_time_ = gezi.get_timestr(int(now_time))
	content = '[{} {} {} {}]{}'.format(score, num_posts, int(now_time) - int(create_time), thre, content)
	val = [forum_id, thread_id, post_id , command_no, create_time_, title, content, uname, forum, uid, ip]
	vals.append(val)

try:
	print 'writting to db count: ' + str(len(vals))
	cur.executemany(sql, tuple(vals))
except Exception, e:
	print e

conn.commit()
cur.close()
conn.close()

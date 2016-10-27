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
 
command_no = '30002'

conn = MySQLdb.connect(host='10.99.89.42',user='root',passwd='root',db='tieba',port=3316)
cur = conn.cursor()
cur.execute('set names gbk')

sql = "delete from dm_monitor where monitor_type = %s"%command_no

cur.execute(sql)

conn.commit()
cur.close()
conn.close()

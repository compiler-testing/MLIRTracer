# -*- coding: utf-8 -*-
# import sys


import os
import subprocess
import sys
import pymysql
import signal
import datetime
import re
import json

sys.path.append('../')
from utils import *
from utils.logger_tool import log
from utils.config import Config
from utils.IR_analyzer import *

def runMLIR(cmd):
    pro = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, universal_newlines=True, encoding="utf-8",preexec_fn=os.setsid)
    try:
        stdout, stderr = pro.communicate(timeout=5)
        return_code = pro.returncode
    except subprocess.TimeoutExpired:
        os.killpg(pro.pid,signal.SIGTERM)
        return_code = 999
    return return_code

def generate_user_cases(conf: Config, seeds_count, mode):
    count=0;
    i = 0

    while(count< conf.count):
        log.info("generating " + str(count)+'/' + str(conf.count))
        target_file = conf.temp_dir + str(count) + '.mlir'
        genrateOpt = "-tosaGen"
        cmd = '%s %s %s -o %s' % (conf.mlirfuzzer_opt, conf.empty_func_file, genrateOpt,target_file)

        i = i+1
        if runMLIR(cmd)==0:
            with open(target_file, 'r') as f:
                content = f.read()
            operations = IRAnalysis(content)
            candidate_lower_pass = ""
            if len(content)>10:
                try:
                    sql = "insert into "+ conf.seed_pool_table + \
                          " (preid,source,operation,content,n, candidate_lower_pass) " \
                          "values ('%s','%s','%s','%s','%s','%s')" \
                          % \
                          (0,'G',operations,content, 0, candidate_lower_pass)
                    dbutils.db.executeSQL(sql)
                    count = count+ 1
                except Exception as e:
                    log.error('sql error', e)
                os.remove(target_file)
            else:
               log.info("Insufficient seed length")


def create_new_table(conf: Config):
    with open('./conf/init.sql', 'r',encoding="utf-8") as f:
        sql = f.read().replace('seed_pool_table', conf.seed_pool_table) \
            .replace('result_table', conf.result_table) \
            .replace('report_table', conf.report_table)
    try:
        dbutils.db.connect_mysql()
        sql_list = sql.split(';')[:-1]
        for item in sql_list:
            dbutils.db.cursor.execute(item)
        dbutils.db.db.commit()
        print("database init success!")
    except pymysql.Error as e:
        print("SQL ERROR:=======>", e)
    finally:
        dbutils.db.close()



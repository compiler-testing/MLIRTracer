# -*- coding: utf-8 -*-

import random
import datetime
import subprocess
import sys
import json
from time import time
import Levenshtein
from random import randint
import os
import numpy as np
import re
import difflib
import signal  
import time

sys.path.append('../')
sys.path.append('./')
from utils.config import Config
from utils import *
from utils.dbutils import myDB
from utils.logger_tool import log
from utils.IR_analyzer import *
from fuzz.pass_enum import *

from fuzz.reporter import *





class Fuzz:
    def __init__(self, config: Config,reporter: Reporter):
        self.config = config
        self.reporter = reporter

    def update_nindex(self, sid, reset=False):
        if reset:
            sql = "update " + self.config.seed_pool_table + " set n = 0 where sid = '%s'" % sid
            dbutils.db.executeSQL(sql)
        else:
            sql = "update " + self.config.seed_pool_table + " set n = n +1 where sid = '%s'" % sid
            dbutils.db.executeSQL(sql)

    def analysis_and_save_seed(self,record_model,output_file):
        sid = record_model.sid
        phase = record_model.phase

        with open(output_file, 'r',encoding='utf-8') as f:
            trans_mlir = f.read()

        ops = IRAnalysis(trans_mlir)
        candidate_lower_pass = ""
        try:
            sql = "insert into " + self.config.seed_pool_table + \
                  " (preid,source,operation, content,n, candidate_lower_pass) " \
                  "values ('%s','%s','%s','%s','%s','%s')" \
                  % \
                  (sid,phase,ops,trans_mlir, 0, candidate_lower_pass)
            dbutils.db.executeSQL(sql)
            self.update_nindex(sid)
        except Exception as e:
            log.error('sql error', e)

    def FuzzingSave(self,record_model):
        sid =record_model.sid
        passes = record_model.passes.replace("'", "")
        returncode = record_model.returncode
        stderr = record_model.stderr.replace("'", "")
        mlir = record_model.mlir
        phase = record_model.phase
        time = record_model.time
        duration = record_model.duration

        ops = IRAnalysis(mlir)

        try:
            sql = "insert into  " + self.config.result_table + \
                  " (sid, content,phase,operation,passes,returnCode,stdout,stderr,duration,datetime) " \
                  "values ('%s','%s','%s','%s','%s','%s','%s','%s','%s','%s')" \
                  % \
                  (sid, mlir, phase,ops, passes, returncode, "", stderr, duration, time)
            dbutils.db.executeSQL(sql)

        except Exception as e:
            log.error('sql error', e)

    def bugReport(self,record_model):
        passes = record_model.passes
        returncode = record_model.returncode
        stderr = record_model.stderr.replace("'", "")
        mlir = record_model.mlir
        phase = record_model.phase
        time = record_model.time


        bug_info = self.reporter.getCrashInfo(stderr)


        if bug_info in self.config.bugs_info:
            print("It's not a new bug!")
        else:
            try:
                sql = "insert into " + self.config.report_table + \
                      " (new,stack,phase,datetime,stderr,returnCode,mlirContent,passes) " \
                      " values('%s','%s','%s','%s','%s','%s','%s','%s')" \
                      % \
                      (0, bug_info, phase, time, stderr.replace("'", "\'"), returncode, mlir, passes)
                dbutils.db.executeSQL(sql)
                self.config.bugs_info.append(bug_info)
            except Exception as e:
              log.error('sql error', e)

    def fixlowerpass(self,singlePass):
        segs = singlePass.split(" -")
        if len(segs)==1:
            return singlePass

        pass_str = singlePass.replace(" -pass-pipeline", " --pass-pipeline")

        insert = " | " + self.config.mlir_opt
        if "-pass-pipeline" not in segs[0]:
            pass_str = pass_str.replace(" --pass-pipeline", insert + " --pass-pipeline")

        pass_list = []
        for pass_ in pass_str.split(insert):
            pass_ = pass_.lstrip()
            if "pass-pipeline" in pass_ and " -" in pass_:
                first_dash_index = pass_.find(' -')

                if first_dash_index != -1:
                    modified_string = pass_[:first_dash_index] + insert+ " " + pass_[first_dash_index:]
                pass_ = modified_string

            pass_list.append(pass_)
            i_str = insert +" "
        return i_str.join(pass_list)

    def runMLIR(cmd):
        pro = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, universal_newlines=True, encoding="utf-8",preexec_fn=os.setsid)
        try:
            stdout, stderr = pro.communicate(timeout=30)
            return_code = pro.returncode
        except subprocess.TimeoutExpired:
            os.killpg(pro.pid,signal.SIGTERM)
            stdout = ""
            stderr = "timeout, kill this process"
            return_code = 99
        except UnicodeDecodeError:
            stderr = "UnicodeDecodeError: 'utf-8' codec can't decode byte 0x8b in position 3860: invalid start byte"
            return_code = 88

        return return_code,stderr

    def MLIRTest(self,sid,mlir_content,seed_file, output_file,pass_str,phase):
        self.config.Iter += 1
        if "-pass-pipeline" in pass_str:
            pass_str = self.fixlowerpass(pass_str)

        cmd = "{} {} {} -o {}".format(self.config.mlir_opt, seed_file, pass_str, output_file)

        start_time = int(time.time() * 1000)
        returncode,stderr = Fuzz.runMLIR(cmd)
        end_time = int(time.time() * 1000)
        duration = int(round(end_time - start_time))

        log.info("#Testing :" + str(self.config.Iter) + "  #Bugs :" + str(len(self.config.bugs_info)) + "  Phase: " + phase )

        now = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
        record_model = recordObject(sid=sid,returncode=returncode, passes=pass_str, stderr=stderr,
                                    mlir=mlir_content, phase=phase,time=now,duration=duration)

        self.FuzzingSave(record_model)
        if returncode == 0 and os.path.isfile(output_file):
            diff = "diff {} {} > /dev/null 2>&1".format(seed_file, output_file)
            result = os.system(diff)
            if result != 0 :
                self.analysis_and_save_seed(record_model,output_file)
        elif returncode > 130:
            self.bugReport(record_model)


    def select_seed(self,Num):
        sql = "select * from " + self.config.seed_pool_table + " where n < '%s'" % 1 + "ORDER BY rand() limit %s"  % Num
        seed_types = dbutils.db.queryAll(sql)
        return seed_types

    def select_target_seed(self,dia,Num):
        sql = "select * from " + self.config.seed_pool_table + " where n < '%s' and operation like '%%" % 1 + dia + "%%' ORDER BY rand() limit %s" % Num
        seed_types = dbutils.db.queryAll(sql)
        return seed_types

    def process(self):
        conf = self.config
        seed_file = conf.temp_dir + "seed" + ".mlir"

        if not os.path.exists(seed_file):
            os.system(r"touch {}".format(seed_file))

        output_file = conf.temp_dir + "output" + ".mlir"
        output1_file = conf.temp_dir + "lower" + ".mlir"
        output2_file = conf.temp_dir + "mutate" + ".mlir"

        proi_lower = [["tosa", "linalg", [
            "-tosa-optional-decompositions -pass-pipeline=\"builtin.module(func.func(tosa-to-linalg-named))\""]],
                      ["tosa", "linalg", ["-tosa-optional-decompositions -pass-pipeline=\"builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg))\""]],
                      ["tosa", "linalg", ["-tosa-optional-decompositions -tosa-to-tensor",
                          "-tosa-optional-decompositions -tosa-to-scf"]],
                      ["linalg", "affine", ["-linalg-bufferize -convert-linalg-to-parallel-loops"]],
                      ["linalg", "affine", ["-linalg-bufferize -convert-linalg-to-affine-loops"]],
                      ["linalg", "affine", ["-linalg-bufferize -convert-linalg-to-loops"]],
                      ["affine.for", "scf", ["-affine-parallelize"]],
                      ["affine", "scf", ["-lower-affine",
                                         "-affine-super-vectorize=\"virtual-vector-size=128 test-fastest-varying=0 vectorize-reductions=true\""]],
                      ["vector", "scf", ["-convert-vector-to-scf"]],
                      ["vector", "gpu", ["-convert-vector-to-gpu"]],
                      ["affine.for", "gpu", [
                          "-pass-pipeline=\"builtin.module(func.func(convert-affine-for-to-gpu{gpu-block-dims=1 gpu-thread-dims=0}))\""]],
                      ["scf.parallel", "gpu", ["-gpu-map-parallel-loops -convert-parallel-loops-to-gpu"]],
                      ["scf.parallel", "async",
                       ["--async-parallel-for=num-workers=-1", "--async-parallel-for =async-dispatch=true",
                        "--async-parallel-for async-dispatch=false", "--async-parallel-for"]],
                      ["gpu", "async", ["-gpu-kernel-outlining -gpu-async-region"]],
                      ["async", "async", ["-async-to-async-runtime -async-runtime-ref-counting",
                                          "-async-to-async-runtime -async-runtime-ref-counting-opt "]]
                      ]


        for r in proi_lower:
            seeds = self.select_target_seed(r[0], 50)
            print(r[0])
            for seed in seeds:
                sid = seed[0]
                ops, mlir_content, n, lowerPass = seed[-4:]

                random_number = random.random()
                if ops != {} and random_number < 0.3:
                    print(ops)
                    IR_ops = []
                    for key, value in json.loads(ops).items():
                        IR_ops.extend(value)

                    Re_ops = []
                    for key, value in conf.ops_mutate.items():
                        Re_ops.append(key)

                    set1 = set(IR_ops)
                    set2 = set(Re_ops)
                    common_ops = list(set1.intersection(set2))

                    mutate_mlir = mlir_content
                    for op in common_ops:
                        random_op = random.choice(conf.ops_mutate[op])
                        print(op, " ", random_op)
                        count = int((mlir_content.count(op) + 1) / 2)
                        mutate_mlir = mutate_mlir.replace(op, random_op, count)
                    mlir_content = mutate_mlir

                with open(seed_file, 'w', encoding='utf-8') as f:
                    f.write(mlir_content)

                if ops != {} and random_number > 0.7:
                    IR_ops = []
                    for key, value in json.loads(ops).items():
                        IR_ops.extend(value)
                    if "tosa" in IR_ops or r[0] == "linalg":
                        cmd = "{} {} {} -o {}".format(conf.mlirfuzzer_opt, seed_file, "-Mix", output2_file)
                        self.verifyMutate(cmd)
                        if os.path.exists(output2_file):
                            os.system("mv " + output2_file + " " + seed_file)

                selected_pass = random.choice(r[-1])
                self.MLIRTest(sid, mlir_content, seed_file, output1_file, selected_pass, "L")

                for i in range(5):
                    trans_pass = random.sample(conf.opts, conf.pass_num)
                    selected_pass = ' '.join(trans_pass)
                    self.MLIRTest(sid, mlir_content, seed_file, output_file, selected_pass,"T")

        seeds = self.select_seed(200)
        for selected_seed in seeds:
            sid = selected_seed[0]
            ops, mlir_content, n, lowerPass = selected_seed[-4:]

            with open(seed_file, 'w',encoding='utf-8') as f:
                f.write(mlir_content)

            candidate_pass = IRaware_pass_selection(self.config,ops)
            if candidate_pass == []:
                continue
            selected_pass = random.choice(candidate_pass)
            self.MLIRTest(sid, mlir_content, seed_file, output1_file, selected_pass,"L")

            for i in range(5):
                trans_pass = random.sample(conf.opts, conf.pass_num)
                selected_pass = ' '.join(trans_pass)
                self.MLIRTest(sid, mlir_content, seed_file, output_file, selected_pass,"T")








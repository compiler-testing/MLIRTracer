# -*- coding: utf-8 -*-

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


def IRAnalysis(content):
    dialectSet = {'acc', 'affine', 'amdgpu', 'amx', 'arith', 'arm_neon', 'arm_sve', 'arm_sme', 'async',
                  'bufferization',
                  'cf', 'complex', 'dlti', 'emitc', 'func', 'gpu', 'index', 'irdl', 'linalg', 'llvm', 'math',
                  'memref',
                  'mesh', 'ml_program', 'mpi', 'nvgpu', 'nvvm', 'omp', 'pdl_interp', 'pdl', 'polynomial', 'quant',
                  'rocdl',
                  'scf', 'shape', 'sparse_tensor', 'tensor', 'ub', 'vcix', 'vector', 'x86vector', 'xegpu', 'spriv',
                  'tosa',
                  'transform'}


    diaList = []
    for dialect in dialectSet:
        if dialect + '.' in content:
            diaList.append(dialect)

    dia_dict = {}
    for dialect in diaList:
        dia_dict[dialect] = []
        ops = re.findall(dialect + r'\.\w+', content)
        opNameSet = set()
        for op in ops:
            opNameSet.add(op)
        for opName in opNameSet:
            dia_dict[dialect].append(opName)

    json_result = json.dumps(dia_dict, indent=4)
    return json_result


def IRaware_pass_selection(config, IR_ops):
    pass_list = []
    try:
        IR_ops = json.loads(IR_ops)
    except json.decoder.JSONDecodeError:
        print("JSONDecodeError: No valid JSON object could be decoded from the string.")
        return pass_list
    
    dependency = config.dependency_lower


    for key in IR_ops.keys():
        if key == "func" or key not in dependency.keys():
            continue
        for d_ops in dependency[key]["OPS"]:
            d_ops = [s.replace("-", "") for s in d_ops]
            has_common = bool(set(d_ops) & set(IR_ops[key]))
            if has_common:
                index = dependency[key]["OPS"].index(d_ops)
                pass_ = dependency[key]["PASS"][index]
                pass_list.append(pass_)
        if key == "linalg":
            pass_list.extend(dependency[key]["PASS"])
    return pass_list
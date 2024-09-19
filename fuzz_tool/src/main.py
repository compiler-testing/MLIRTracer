# -*- coding: utf-8 -*-

import argparse
import sys
import time
import json
import datetime
from utils.config import Config
from utils import *
from fuzz.reporter import *


import os
from utils.logger_tool import log
reported_errors = []

def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--opt', required=True,
                            choices=['generator', 'fuzz'])
    arg_parser.add_argument('--sqlName',required=True)
    return arg_parser.parse_args(sys.argv[1:])

import re

def main():
    args = get_args()
    config_path = './conf/conf.yml'
    conf = Config(config_path,args)
    
    logger_tool.get_logger()
    dbutils.db = dbutils.myDB(conf)
    if args.opt == 'generator':
        from utils.tosaGen import generate_user_cases,create_new_table
        # initialize database
        create_new_table(conf)
        # generate tosa graphs
        generate_user_cases(conf, conf.count,args.mode)


    elif args.opt == 'fuzz':
        from fuzz.fuzz import Fuzz
        reporter = Reporter(conf)
        fuzzer = Fuzz(conf,reporter)

        start = datetime.datetime.now()
        end = start + datetime.timedelta(minutes=conf.run_time)
        st= start.timestamp()
        nt = st
        while (nt-st<conf.run_time):
            now = datetime.datetime.now()
            nt= now.timestamp()
            if args.debug != '0':
                fuzzer.debug()
                break
            fuzzer.process()
            if now.__gt__(end):
                break
        print("time out!!!")


if __name__ == '__main__':
    main()

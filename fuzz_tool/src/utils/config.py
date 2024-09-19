import yaml
import os
import time
import json

def load_conf(conf_path):
    with open(conf_path, 'r',encoding='utf-8') as f:
        content = f.read()
    conf = yaml.load(content, Loader=yaml.FullLoader)
    return conf


class Config:
    def __init__(self, conf_path,args):
        sqlName = args.sqlName
        flag = args.opt

        conf = load_conf(conf_path)
        common = conf['common']
        database = conf['database']
        generator = conf['generator']
        fuzz = conf['fuzz']
        report = conf['report']

        if path_mlir_opt == "default":
            self.mlir_opt = common['opt_executable']
        else:
            self.mlir_opt = path_mlir_opt


        self.host = database['host']
        self.port = database['port']
        self.username = database['username']
        self.passwd = database['passwd']

        self.db = database['db']


        label = sqlName
        self.seed_pool_table = 'seed_pool_' + label
        self.result_table = 'result_' + label
        self.report_table = 'report_' + label

        self.mlirfuzzer_opt = common['project_path'] + generator['generator_executable']
        self.empty_func_file = generator['empty_func_file']
        self.count = generator['count']


        self.run_time = fuzz['run_time']
        self.pass_num = fuzz['pass_num']
        self.temp_dir = common['project_path']+ fuzz['temp_dir']

        self.bugs_info = []
        self.mutate_flag = fuzz['mutate_flag']

        if flag == "fuzz":

            passes = []
            with open(fuzz['opt_file'], 'r') as f:
                for line in f.readlines():
                    passes.append(line.replace('\n', ''))
            self.opts = passes


            passes = []
            with open(fuzz['lower_file'], 'r') as f:
                for line in f.readlines():
                    passes.append(line.replace('\n', ''))
            self.opts_lower = passes

            self.temp_dir = self.temp_dir+"temp."+ str(hash(time.time())) +"/"
            if not os.path.exists(self.temp_dir):
                os.makedirs(self.temp_dir)

            passes = []
            with open(fuzz['lower_dependency'], 'r') as f:
                jsonData = json.load(f)
            self.dependency_lower = jsonData

            with open(fuzz['ops_mutate'], 'r') as f:
                self.ops_mutate = json.load(f)

        self.Iter = 0



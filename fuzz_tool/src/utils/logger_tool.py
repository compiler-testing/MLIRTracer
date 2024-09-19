import logging
import time
import os
log=None


def get_logger():
    global log
    if log is not None :
        return log
    logger = logging.getLogger()
    curdir = os.getcwd()
    log_dir = os.path.dirname(curdir) + '/logs/'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    print(log_dir)
    if not logger.handlers:

        logger.setLevel(logging.DEBUG)
        uuid_str = time.strftime("%Y-%m-%d-%H", time.localtime())
        log_file = '%s%s.txt' % (log_dir, uuid_str)
        fh = logging.FileHandler(filename=log_file, encoding="utf8")
        sh = logging.StreamHandler()

        fmt = logging.Formatter(fmt="[%(asctime)s]- %(levelname)s (%(lineno)s): %(message)s ", datefmt="%Y/%m/%d %H:%M:%S")

        fh.setFormatter(fmt)
        sh.setFormatter(fmt)

        sh.setLevel(logging.INFO)
        logger.addHandler(fh)
        logger.addHandler(sh)
        log = logger

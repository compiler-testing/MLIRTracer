import pymysql
from utils.config import Config
from utils.logger_tool import log

db=None


class myDB:

    def __init__(self, config: Config):
        self.host = config.host
        self.username = config.username
        self.password = config.passwd
        self.port = config.port
        self.database = config.db

    def connect_mysql(self):

        self.db = pymysql.connect(host=self.host, port=self.port, user=self.username, password=self.password,
                                  database=self.database)
        self.cursor = self.db.cursor()

    def executeSQL(self, sql):
        try:
            self.connect_mysql()
            self.cursor.execute(sql)
            self.db.commit()
        except Exception as e:
            log.info("SQL ERROR:=======>", e)
            log.info("wrong SQL:=======>", sql)
        finally:
            self.close()

    def queryAll(self, sql):

        self.connect_mysql()
        self.cursor.execute(sql)
        data = self.cursor.fetchall()
        self.close()
        return data

    def queryOne(self, sql):
        self.connect_mysql()
        self.cursor.execute(sql)
        data = self.cursor.fetchone()
        self.close()
        return data

    def close(self):
        self.cursor.close()
        self.db.close()

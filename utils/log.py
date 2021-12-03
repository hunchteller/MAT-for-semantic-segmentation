import logging
import os

class Log(object):
    def __init__(self, filename='log', filemode='a+', level='info'):
        levels = {'debug':logging.DEBUG, 'info':logging.INFO, 'warning':logging.WARNING, 'error':logging.ERROR}
        if level in levels.keys():
            level = levels[level]
        else:
            raise KeyError('only accept [debug, info, warning, error]')
        
        self.log = logging.getLogger(__name__)
        self.log.setLevel(level)
        handler = logging.FileHandler(filename, filemode, encoding='utf-8')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.log.addHandler(handler)
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(formatter)
        self.log.addHandler(console)
        self.info('\n======================================\n\n======================================')

    def debug(self, message):
        self.log.debug(message)
    
    def info(self, message):
        self.log.info(message)

    def warning(self, message):
        self.log.warning(message)

    def error(self, message):
        self.log.error(message)

    def critical(self, message):
        self.log.critical(message)



if __name__ == "__main__":
    logging = Log(level='info')
    logging.debug('DEBUG 信息')
    logging.info('info 信息')
    logging.warning('warning message')
    logging.error('error message')
    logging.critical('critical message')


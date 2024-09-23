import sys
import logging
sys.path.insert(0,'C:/Users/admin/Desktop/AIServer')
def get_logger()->logging.Logger:
    logger=logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # 设置日志级别为DEBUG
    handler = logging.StreamHandler()  # 创建一个StreamHandler
    handler.setLevel(logging.DEBUG)  # 设置handler的日志级别
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # 定义日志格式
    handler.setFormatter(formatter)  # 应用日志格式到handler
    logger.addHandler(handler)  # 将handler添加到logger
    return logger

def Singleton(cls):
    '''
    实现单例
    '''
    _instantce={}

    def _sinleton(*args,**kargs):
        if cls not in _instantce:
            _instantce[cls]=cls(*args, **kargs)
        return _instantce[cls]
import os,sys,re
sys.path.append(r'C:\Users\admin\Desktop\AIServer')
from modules.Intent_routing_center import IntentRoutingCenter
from modules.draw_instruct import DrawInstruct
from utils.utils import get_logger,Singleton

logger=get_logger()

def load_intent_model():
    service_model=IntentRoutingCenter()
    return service_model

def load_draw_instruct_model():
    service_model=DrawInstruct()
    return service_model


@Singleton
class RejectStrategy(object):
    def __init__(self) -> None:
        self.sensitive_words=self.load_reject_words()

    @classmethod
    def load_reject_words(self):
        sensitive_words=[]
        try:
            with open("./resources/sensitive.txt",'r',encoding='utf-8') as f:
                for line in f:
                    logger.info(f"loading reject words from :{os.path.abspath("./resources/sensitive.txt")}")
                    sensitive_words.append(line.strip())
            return sensitive_words
        except Exception as e:
            logger.warning(f"load reject words failed,error:{e}")
    def check_reject(cls,query:str)->bool:
        raise NotImplementedError


class RejectStrategyForDrawScene(RejectStrategy):
    '''
    Function check_reject: return True if there is sensitive words in query
    '''
    def __init__(self) -> None:
        super().__init__()
        self.reject_pattern='|'.join(word for word in self.sensitive_words)

    def check_reject(cls,query:str)->bool:
        if re.search(cls.reject_pattern,query):
            return True
        return False



'''
descripition:配置类
update:2024.9.20
'''
import os
import sys
import json
sys.path.append(r'C:/Users/admin/Desktop/AIServer')
from utils.utils import get_logger,Singleton



@Singleton
class CFG:
    def __init__(self)->None:
        self.logger=get_logger()
        self.intent_keywords_path='./settings/intent_keywords.json'
        self.intent_model_param_path='./PretrainModels/intent_predict'
        self.intent_keywords=None
        self.initial_config()
    def initial_config(self)->None:
        self.logger.info("load settings")
        self.load_intent_keywords()
        self.logger.info("load settings finish")
    def load_intent_keywords(self)->None:
        '''
        从配置文件中读取意图识别关键词。
        当识别到输入中有相关关键词时候，会直接返回意图
        '''
        self.logger.info(f"loading intent keywords setting in {os.path.abspath(self.intent_keywords_path)}")

        if os.path.exists(self.intent_keywords_path):
            with open(self.intent_keywords_path,'r',encoding='utf-8') as file:
                intent_keywords=json.load(file)
                self.intent_keywords=intent_keywords
                return intent_keywords
        else:
            raise ValueError(f"path {os.path.abspath(self.intent_keywords_path)} not exist")
        
    





                
if __name__ == '__main__':
    cfg=CFG()
    cfg.load_intent_keywords()
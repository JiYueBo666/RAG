"""
descripition:意图识别模块，负责将输入路由到不同的处理模块
update:2024.9.20
"""

import os, sys, re
import torch

sys.path.append(r"C:\Users\admin\Desktop\AIServer")
from utils.utils import Singleton
from settings.config import CFG
from models.intent_classification import IntentClassification


class IntentRoutingCenter:
    '''
    意图识别模块，先规则后模型
    '''
    def __init__(self) -> None:
        # 加载意图识别模型
        self.cfg = CFG()
        init_model = self.load_model()
        self.model = [init_model, init_model]
        self.cur_run_model_idx = 0#当前使用的模型索引
        self.cur_run_intent_idx = 0#当前使用的意图模板索引
        self.regular_intent = [self.cfg.intent_keywords,self.cfg.intent_keywords]  
        # 采用关键词判断意图 dict 格式 {'意图1':[keywords1,keywords2.....],'意图2':[.....]}
        self.intent_map = {"绘画": 0, "搜索": 1}
    def load_model(self) -> IntentClassification:
        model = IntentClassification()
        # 加载模型参数
        ""
        ...
        ""
        return model

    def update_model(self) -> IntentClassification:
        """
        用于热更新模型参数
        """
        new_model = self.load_model()  # 加载新模型
        self.model[(self.cur_run_model_idx + 1) % 2] = new_model
        new_cur_run_model_idx = (self.cur_run_model_idx + 1) % 2
        self.cur_run_model_idx = new_cur_run_model_idx
        return

    def update_intent(self) -> dict:
        """
        意图扩充
        """
        new_intent_dict = self.cfg.load_intent_keywords()
        self.regular_intent[(self.cur_run_intent_idx + 1) % 2] = new_intent_dict
        self.cur_run_intent_idx = (self.cur_run_intent_idx + 1) % 2
        return
                       
    def __model_predict(self, query: str) -> int:
        raise NotImplementedError
        # 对用户输入的预处理..,tokenize化
        # ....
        return self.model[self.cur_run_model_idx](query)

    def __regular_predict(self, query:str) -> int:
        '''
        正则表达式的意图识别
        '''
        cur_intent_dict = self.regular_intent[self.cur_run_intent_idx]  # dict
        intent = None
        for key, values in cur_intent_dict.items():
            for value in values:
                if re.search(value, query):
                    intent = key
                    break
        if intent is not None:
            # 映射intent到数字
            """
            这里先写死，应该也需要热更新,同时这里的映射数字需要和模型输出的意图分类值一致.
            """
            intent = self.intent_map[intent]
            return intent
        return None
    def predict(self,query:str)->int:
        intent=self.__regular_predict(query=query)
        if intent is not None:
            return intent
        else:
            intent=self.__model_predict(query=query)
            return intent
        
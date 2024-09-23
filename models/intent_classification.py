import torch,sys,os
import torch.nn as nn
sys.path.append(r'C:\Users\admin\Desktop\AIServer')
from utils.utils import Singleton

@Singleton
class IntentClassification(nn.Module):
    '''
    class for intent predict,return a int value which can be mapped to a intent
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pass

    def predict_intent(self,query:str)->int:
        return -1
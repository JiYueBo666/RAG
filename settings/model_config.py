import sys
sys.path.append(r"C:\Users\admin\Desktop\AIServer")
class LSTMConfig:
    # 将所有属性定义为类变量
    lstm_hidden_size = 256
    token_size = 5561
    entity_type_num = 7
    device='cuda'
    checkpoint_path='./PretrainModels/LSTM/checkpoints.pth'
    max_size=20


    @classmethod
    def hidden_size(cls):
        return cls.lstm_hidden_size
    @classmethod 
    def get_token_size(cls):
        return cls.token_size
    @classmethod
    def get_entity_type_num(cls):
        return cls.entity_type_num
    @classmethod
    def get_device(cls):
        return cls.device
    @classmethod
    def get_param_path(cls):
        return cls.checkpoint_path
    @classmethod
    def get_max_size(cls):
        return cls.max_size

import torch, os
import sys

sys.path.append(r"C:\Users\admin\Desktop\AIServer")
from utils.utils import Singleton, get_logger
from load_modules import RejectStrategyForDrawScene
from models.ner_model import BiLSTM, Tokenizer
from settings.model_config import LSTMConfig
from dialogue_manager import MetaData, DialogueManager
from modules.draw.slot_extract import RegularExtract
from modules.corrector import Corrector

logger = get_logger()


class DrawInstruct:
    def __init__(self) -> None:
        self.reject_helper = RejectStrategyForDrawScene()
        self.lstm_crf = BiLSTM()
        self.tokenizer = Tokenizer()
        self.dialogue_manager = DialogueManager()
        self.corrector = Corrector()
        self.slot_extractor = RegularExtract()
        self.source = [
            "一元环",
            "二元环",
            "三元环",
            "四元环",
            "五元环",
            "六元环",
            "七元环",
            "八元环",
            "苯环",
            "单键",
            "双键",
            "三键",
            "碳碳双键",
            "1元环",
            "2元环",
            "3元环",
            "4元环",
            "5元环",
            "6元环",
            "7元环",
            "8元环",
            "3键",
            "环丙烷",
            "环丁烷",
            "环戊烷",
            "环己烷",
            "环庚烷",
            "环辛烷",
            "继续",
            "清屏",
            "提交",
            "甲基",
        ]

    def init_model(self) -> None:
        self.checkpoint_path = LSTMConfig.checkpoint_path
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            self.lstm_crf.load_state_dict(checkpoint["model_state_dict"])
        else:
            logger.error(f"checkpoint file not found in {self.checkpoint_path}")

    def instruction_analysis(self, query: str) -> dict:
        # check if there is sensitive words
        if self.reject_helper.check_reject(query) == False:
            return {"code": -1, "msg": "sensitive words error"}
        else:
            slot = self.slot_extractor.entity_extract_slot(
                query, self.lstm_crf, self.tokenizer
            )
            slot, new_text = self.corrector.error_correct_pinyin_from_source(
                slot, query, source=self.source
            )
            slot=self.slot_extractor.confirm(slot)
            last_frame=self.dialogue_manager.get_frame()
            if last_frame is not None:
                slot=self.slot_extractor.rebuild(slot,last_frame.slot)
            self.dialogue_manager.add_frame(MetaData(intent=0,slot=slot))
            return slot
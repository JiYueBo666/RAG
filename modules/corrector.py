import Levenshtein
from pypinyin import lazy_pinyin
import sys
sys.path.append(r"C:\Users\admin\Desktop\AIServer")

from utils.utils import get_logger

logger = get_logger()


class Corrector:
    def __init__(self) -> None:
        self.max_workers = 20  # 线程数
        self.threshold = 0.75  # 控制拼音相似度阈值
        self.entity_map = {
            "环丙烷": "3元环",
            "黄金丸": "4元环",
            "环丁烷": "4元环",
            "环雾丸": "5元环",
            "环戊烷": "5元环",
            "缓解完": "6元环",
            "环己烷": "6元环",
            "环庚烷:": "7元环",
            "环辛烷": "8元环",
            "环壬烷": "9元环",
            "甲基": "单键",
        }

    def error_correct_pinyin_from_source(
        self, slot: dict, text: str, source=None
    ) -> dict:
        assert isinstance(slot, dict), "slot 必须是字典"
        candidates = []
        new_text = None
        if source is None:
            for key, value in slot.items():
                if key == "DO" or key == "IO" and value != "None":
                    # 召回
                    candidates = self.edit_distance_recall("".join(lazy_pinyin(value)))
                    if len(candidates) == 0:
                        continue
                    pinyin_sim = self.pinyin_sorted(
                        recall_list=candidates, query_words=value
                    )
                    if pinyin_sim[0] >= self.threshold:
                        if slot[key] in text:
                            new_text = text.replace(slot[key], pinyin_sim[1])
                        slot[key] = pinyin_sim[1]
        else:
            if isinstance(source, str):
                try:
                    with open(source, "r", encoding="utf-8") as f:
                        for line in f:
                            candidates.append(line.strip())
                except ValueError as e:
                    logger.error(f"error correct from source failed,error:{e}")
            elif isinstance(source, list):
                candidates = source
            for key, value in slot.items():
                if key == "DO" or key == "IO" and value != "None":
                    if slot[key] in source:  # 不需要纠错
                        new_text = text
                        continue
                    pinyin_sim = self.pinyin_sorted(
                        recall_list=candidates, query_words=value
                    )
                    if pinyin_sim[0] >= self.threshold:
                        if slot[key] in text:
                            new_text = text.replace(slot[key], pinyin_sim[1])
                        slot[key] = pinyin_sim[1]
        return slot, new_text

    def compute_cqr_ctr(self, user_query_pinyin: str, candidate: str):
        """
        :param user_query_pinyin:
        :param candidate:
        :return:
        """
        candidate_pinyin = "".join(lazy_pinyin(candidate))
        max_length = max(len(user_query_pinyin), len(candidate_pinyin))
        cqr = self.predict_left(user_query_pinyin, candidate_pinyin)
        ctr = self.predict_left(candidate_pinyin, user_query_pinyin)
        edit_distance = Levenshtein.distance(user_query_pinyin, candidate_pinyin)
        edit_similarity = (max_length - edit_distance) / max_length

        return [cqr * ctr * 0.8 + edit_similarity * 0.2, candidate]

    def predict_left(self, q1, q2):
        if len(q1) < 1 or len(q2) < 1:
            return 0
        q1 = set(q1)
        q2 = set(q2)
        numerator = len(q1.intersection(q2))
        denominator = len(q1.union(q2))
        return numerator / denominator

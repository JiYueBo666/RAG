'''
descripition:核心检索模块
'''

import sys,os,copy,json
import faiss
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
sys.path.append(r"C:\Users\admin\Desktop\AIServer")
from models.Embedding import BGEmbedding
from utils.utils import get_logger
from core.vector_base import VecIndex
logger=get_logger()

class Searcher:
    def __init__(self) -> None:
        self.path="./PretrainModels/BGE"
        self.init_model()
        self.vec_search=VectorSearch()


    def init_model(self):
        if os.path.exists(self.path):
            self.vec_model=BGEmbedding(self.path)
        else:
            logger.error(f"model path error,path:{self.path} not found")
            raise FileNotFoundError
    def rank(self, query, recall_result):
        rank_result = []
        for idx in range(len(recall_result)):
            new_sim = self.vec_model.predict_sim(query, recall_result[idx][1])
            rank_item = copy.deepcopy(recall_result[idx])
            rank_item.append(new_sim)
            rank_result.append(copy.deepcopy(rank_item))
        rank_result.sort(key=lambda x: x[3], reverse=True)
        return rank_result

    def search(self, query, nums=3):
        q_vec = self.vec_model.get_embedding(query)
        recall_result = self.vec_search.search(q_vec, nums)
        rank_result = self.rank(query, recall_result)
        return rank_result

#class for search

class VectorSearch:
    def __init__(self):
        self.invert_index = VecIndex()
        self.forward_index = []
        self.INDEX_FOLDER_PATH_TEMPLATE = "data/index/{}"

    def build(self, index_name, index_dim: int = 1024):
        self.index_name = index_name
        self.index_folder_path = self.INDEX_FOLDER_PATH_TEMPLATE.format(index_name)
        if not os.path.exists(self.index_folder_path) or not os.path.isdir(
            self.index_folder_path
        ):
            os.mkdir(self.index_folder_path)

        self.invert_index = VecIndex()
        self.invert_index.build(index_dim)

    def insert(self, vec, doc):
        self.invert_index.insert(vec)
        self.forward_index.append(doc)

    def save(self):
        with open(
            self.index_folder_path + "/forward_index.txt", "w", encoding="utf8"
        ) as f:
            for data in self.forward_index:
                f.write("{}\n".format(json.dumps(data, ensure_ascii=False)))

        self.invert_index.save(self.index_folder_path + "/invert_index.faiss")

    def load(self, index_name):
        self.index_name = index_name
        self.index_folder_path = self.INDEX_FOLDER_PATH_TEMPLATE.format(index_name)

        self.invert_index = VecIndex()
        self.invert_index.load(self.index_folder_path + "/invert_index.faiss")

        self.forward_index = []
        with open(self.index_folder_path + "/forward_index.txt", encoding="utf8") as f:
            for line in f:
                self.forward_index.append(json.loads(line.strip()))

    def search(self, vecs, nums=5):
        search_res = self.invert_index.search(vecs, nums)
        recall_list = []
        for idx in range(nums):
            # recall_list_idx, recall_list_detail, distance
            recall_list.append(
                [
                    search_res[1][0][idx],
                    self.forward_index[search_res[1][0][idx]],
                    search_res[0][0][idx],
                ]
            )
        return recall_list


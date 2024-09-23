import numpy as np
import faiss
from typing import List
class VecIndex:
    '''
    数据持久化，存储到向量数据库
    存储维度为1024维
    '''
    def __init__(self) -> None:
        self.index = ""

    def build(self, index_dim: int=1024):
        description = "HNSW64"
        mesure = faiss.METRIC_L2
        self.index = faiss.index_factory(index_dim, description, mesure)

    def insert(self, vectors:np.ndarray):
        self.index.add(vectors)

    def batch_insert(self, vectors: List[List[float]]):
        self.index.add(vectors)

    def load(self, path: str):
        self.index = faiss.read_index(path)

    def save(self, save_path: str) -> None:
        faiss.write_index(self.index, save_path)

    def search(self, query: List[float], k: int = 5) -> List[float]:
        return self.index.search(query, k)
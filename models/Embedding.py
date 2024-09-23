import torch
import os, sys

sys.path.append(r"C:\Users\admin\Desktop\AIServer")
import numpy as np
from transformers import AutoModel, AutoTokenizer
from utils.utils import get_logger
logger=get_logger(__name__)

class BaseEmbedding:
    """
    Base class for all embeddings
    """

    def __init__(self, path: str):
        self.path = path

    def get_embedding(self, text: str) -> list[float]:
        raise NotImplementedError

    @classmethod
    def cosine_similarity(cls, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        calculate cosine similarity between two vectors
        """
        # 计算两个向量的余弦相似度
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)

        # 避免除以零
        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return float(similarity)


class BGEmbedding(BaseEmbedding):
    """
    使用BGE算法进行句子嵌入的类
    :args: path-- 模型文件夹的路径
    """

    def __init__(self, path: str):
        super().__init__(path)
        self.embedding_model_path = path  # 嵌入模型的路径
        if os.path.exists(self.embedding_model_path):
            self.bge_model = AutoModel.from_pretrained(self.path)  # 加载预训练模型
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.path
            )  # 加载预训练的分词器
        else:
            raise ValueError("model path is not set")  # 如果路径不存在，抛出异常
        self.model_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # 设置模型运行的设备
        self.bge_model.to(self.model_device)  # 将模型加载到指定设备
        self.warmup_iter = 2  # 预热迭代次数
        self.warmup()  # 执行预热

    def predict_sim(self, text1: str, text2: str) -> float:
        # 获取两个文本的嵌入向量
        embedding1 = self.get_embedding(text1)
        embedding2 = self.get_embedding(text2)
        # 计算余弦相似度
        similarity = self.cosine_similarity(embedding1[0], embedding2[0])
        return similarity

    def get_embedding(self, text: str) -> np.ndarray:
        """
        获取文本的嵌入向量
        :return:
            np.array, shape is (1,1024)
        """
        encoded_input = self.get_encode_input(text)  # 获取编码后的输入
        # 计算token嵌入
        with torch.no_grad():
            model_output = self.bge_model(**encoded_input)
        # 平均池化 - 考虑注意力掩码以正确平均
        attention_mask = encoded_input["attention_mask"]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(model_output.last_hidden_state.size())
            .float()
        )
        sum_embeddings = torch.sum(
            model_output.last_hidden_state * input_mask_expanded, 1
        )
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        sentence_embeddings = sum_embeddings / sum_mask
        # 归一化嵌入
        sentence_embeddings = torch.nn.functional.normalize(
            sentence_embeddings, p=2, dim=1
        )
        return sentence_embeddings.cpu().numpy()

    def get_encode_input(self, text: str) -> dict:
        """
        tokenizer
        """
        encoded_input = self.tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        )
        # 移动到计算设备
        encoded_input = {k: v.to(self.model_device) for k, v in encoded_input.items()}

        return encoded_input

    @property
    def device(self) -> torch.device:
        return self.model_device

    def warmup(self) -> None:
        """
        模型预热
        """
        logger.info("Embedding Model warming up..。")
        for _ in range(self.warmup_iter):
            self.get_embedding("hello world")  # 使用示例文本进行预热
        logger.info(f"type of embedding is :[np.ndarray]")
        logger.info(f"shape  of embedding is :[(1,1024)]")
        logger.info(f"use device is :[{self.device}]")
        logger.info("Embedding Model warming up finish.")
        return

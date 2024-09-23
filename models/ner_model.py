import torch, os
import torch.nn as nn
from torchcrf import CRF
from settings.model_config import LSTMConfig
from utils.utils import Singleton, get_logger

logger = get_logger()


@Singleton
class BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = LSTMConfig.hidden_size()
        self.entity_type_num = LSTMConfig.get_entity_type_num()
        self.token_size = LSTMConfig.get_token_size()
        self.embedding_layer = nn.Embedding(self.token_size, self.hidden_size)

        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Linear(self.hidden_size * 2, self.entity_type_num)
        self.crf_layer = CRFLayer(num_tags=self.entity_type_num)
        self.drop = nn.Dropout(0.5)

    def forward(
        self,
        x: torch.Tensor,
        senquence_length: torch.Tensor,
        label: torch.Tensor = None,
    ):
        x = self.embedding_layer(x)
        output_sequence, (_, _) = self.lstm(x)  # [batch,seq_len,hidden_size*2]
        lstm_features = self.drop(output_sequence)
        lstm_logits = self.fc(lstm_features)
        mask = self.crf_layer.create_mask(
            senquence_length, max_length=lstm_logits.shape[1]
        )
        if label is not None:
            loss = -self.crf_layer.crf(emissions=lstm_logits, tags=label, mask=mask)
            return loss
        else:
            output = self.crf_layer.crf.decode(emissions=lstm_logits, mask=mask)
            return output


class CRFLayer(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.crf = CRF(num_tags=num_tags, batch_first=True)
        self.device = LSTMConfig.get_device()

    def create_mask(self, sequence_length: torch.Tensor, max_length: int):
        batch_size = sequence_length.shape[0]
        arange = torch.arange(max_length).unsqueeze(0).expand(batch_size, -1)
        if self.device == "cuda":
            arange = arange.to(sequence_length.device)
        # 将范围张量扩展成 [batch_size, max_sequence_length] 的形状，
        # 并与每个序列的实际长度进行比较，小于实际长度的位置为True，否则为False
        mask = arange < sequence_length.unsqueeze(1)
        return mask


@Singleton
class Tokenizer:
    def __init__(self) -> None:
        self.token = {}
        self.schem = {"S": 0, "M": 1, "D": 2, "H": 3, "R": 4, "T": 5, "O": 6}

        self.token_id_path = self.Config.lstm_tokenize_path
        self.__get_token()

    def __get_token(self):
        if os.path.exists(self.token_id_path):
            with open(self.token_id_path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip().split(" ")
                    self.token[line[0]] = int(line[1])
        else:
            logger.error(f"token file not found in {self.token_id_path}")

    def tokenize(self, query: str):
        query = [self.token.get(i, 0) for i in query]
        # The true length of the initial query,instead of padding sequence
        true_length = len(query)
        if len(query) < LSTMConfig.max_size():
            query += [0] * (LSTMConfig.max_size() - len(query))
        elif len(query) >= LSTMConfig.max_size():
            query = query[: LSTMConfig.max_size()]
        return torch.tensor([query]), torch.tensor([true_length])

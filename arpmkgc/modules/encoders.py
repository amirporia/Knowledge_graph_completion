"""
Encoder towers implementing Sec 4.2's E0 and E1:

    q  = E0(h, r)          in R^d      (query encoding, no tail)
    a_i = E0(h_i, r, t_i)   in R^d      (anchor encoding, full triple)
    e_t = E1(t)             in R^d      (entity encoding)

E0 is a single BERT tower reused for both the query and the anchor role
(they differ only in what text is fed in, matching the proposal's "E0:
Query and anchor-enhanced encoder" role). E1 is a second, independently
initialized BERT tower for entities.
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


def pool_hidden_states(pooling: str, last_hidden_state: torch.Tensor,
                       attention_mask: torch.Tensor) -> torch.Tensor:
    if pooling == "cls":
        pooled = last_hidden_state[:, 0, :]
    elif pooling == "mean":
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1e-4)
        pooled = summed / count
    elif pooling == "max":
        mask = attention_mask.unsqueeze(-1).expand_as(last_hidden_state).bool()
        masked_states = last_hidden_state.masked_fill(~mask, -1e4)
        pooled = masked_states.max(dim=1).values
    else:
        raise ValueError(f"Unknown pooling mode: {pooling}")
    return nn.functional.normalize(pooled, dim=-1)


class BertTower(nn.Module):
    """A single BERT encoder + pooling head."""

    def __init__(self, pretrained_model: str, pooling: str = "mean", dropout: float = 0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(pretrained_model)
        self.bert = AutoModel.from_pretrained(pretrained_model)
        if getattr(self.bert, "pooler", None) is not None:
            self.bert.pooler = None
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        hidden = self.dropout(outputs.last_hidden_state)
        return pool_hidden_states(self.pooling, hidden, attention_mask)


class DualTowerEncoder(nn.Module):
    """Wraps the E0 (query/anchor) and E1 (entity) towers."""

    def __init__(self, pretrained_model: str, pooling: str = "mean", dropout: float = 0.1,
                 tie_encoders: bool = False):
        super().__init__()
        self.e0 = BertTower(pretrained_model, pooling, dropout)
        self.e1 = BertTower(pretrained_model, pooling, dropout)
        if tie_encoders:
            self.e1.load_state_dict(self.e0.state_dict())

    @property
    def hidden_size(self) -> int:
        return self.e0.hidden_size

    def encode_query(self, batch: dict) -> torch.Tensor:
        return self.e0(batch["input_ids"], batch["attention_mask"], batch["token_type_ids"])

    def encode_anchors(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                       token_type_ids: torch.Tensor) -> torch.Tensor:
        """input_ids etc. shaped [B, NA, L] -> flattens to [B*NA, L], encodes
        with E0, reshapes back to [B, NA, d]."""
        b, na, seq_len = input_ids.shape
        flat_ids = input_ids.reshape(b * na, seq_len)
        flat_mask = attention_mask.reshape(b * na, seq_len)
        flat_type = token_type_ids.reshape(b * na, seq_len)
        flat_out = self.e0(flat_ids, flat_mask, flat_type)
        return flat_out.reshape(b, na, -1)

    def encode_entity(self, batch: dict) -> torch.Tensor:
        return self.e1(batch["input_ids"], batch["attention_mask"], batch["token_type_ids"])

"""
Encoder towers implementing Sec 4.2's E0 and E1:

    q  = E0(h, r)          in R^d      (query encoding, no tail)
    a_i = E0(h_i, r, t_i)   in R^d      (anchor encoding, full triple)
    e_t = E1(t)             in R^d      (entity encoding)

E0 is a single BERT tower reused for both the query and the anchor role
(they differ only in what text is fed in, matching the proposal's "E0:
Query and anchor-enhanced encoder" role). E1 is a second, independently
initialized BERT tower for entities.

Performance note (not part of the proposal's architecture -- purely an
engineering optimization): encoding the candidate anchor pool is the
dominant cost of a training step, since it runs `candidate_budget` full
BERT forward passes per example versus 1 each for the query/head/tail. Three
knobs below reduce that cost/memory footprint *without changing any score
that gets computed*:

  - `deduplicate_anchors` (default True): identical (h_i, r, t_i) anchors
    within the same flattened [B*NA] batch (e.g. two queries with the same
    relation drawing overlapping global candidates) are encoded once and the
    result is broadcast back -- exact, not approximate.
  - `anchor_encode_chunk_size`: caps how many anchors are run through BERT
    in a single forward call, trading a little wall-clock time for a lower
    peak-memory footprint (mathematically identical output, just computed
    in smaller pieces).
  - `use_gradient_checkpointing`: recomputes anchor-encoder activations
    during the backward pass instead of storing them, trading ~20-30% extra
    compute for a large activation-memory reduction. Exact gradients, no
    change to the loss surface. Training-time only (no effect during eval).
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers import AutoConfig, AutoModel

from ..config import ARPMConfig


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

    def __init__(self, config: ARPMConfig):
        super().__init__()
        self.e0 = BertTower(config.pretrained_model, config.pooling, config.dropout)
        self.e1 = BertTower(config.pretrained_model, config.pooling, config.dropout)
        if config.tie_encoders:
            self.e1.load_state_dict(self.e0.state_dict())

        self.deduplicate_anchors = config.deduplicate_anchors
        self.anchor_encode_chunk_size = config.anchor_encode_chunk_size
        self.use_gradient_checkpointing = config.use_gradient_checkpointing

    @property
    def hidden_size(self) -> int:
        return self.e0.hidden_size

    def encode_query(self, batch: dict) -> torch.Tensor:
        return self.e0(batch["input_ids"], batch["attention_mask"], batch["token_type_ids"])

    def encode_anchors(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                       token_type_ids: torch.Tensor) -> torch.Tensor:
        """input_ids etc. shaped [B, NA, L] -> flattens to [B*NA, L], encodes
        with E0 (deduplicated/chunked/checkpointed per config), reshapes
        back to [B, NA, d]. Output is numerically identical to the naive
        one-call-per-anchor version regardless of which of these knobs are
        enabled -- they only change how the computation is scheduled."""
        b, na, seq_len = input_ids.shape
        flat_ids = input_ids.reshape(b * na, seq_len)
        flat_mask = attention_mask.reshape(b * na, seq_len)
        flat_type = token_type_ids.reshape(b * na, seq_len)

        if self.deduplicate_anchors:
            flat_out = self._encode_deduplicated(flat_ids, flat_mask, flat_type)
        else:
            flat_out = self._run_e0(flat_ids, flat_mask, flat_type)

        return flat_out.reshape(b, na, -1)

    def _encode_deduplicated(self, flat_ids: torch.Tensor, flat_mask: torch.Tensor,
                             flat_type: torch.Tensor) -> torch.Tensor:
        n = flat_ids.size(0)
        unique_ids, inverse = torch.unique(flat_ids, dim=0, return_inverse=True)
        num_unique = unique_ids.size(0)

        # Any original row mapping to a given unique group has identical
        # (ids, mask, type_ids) -- same text tokenizes identically -- so a
        # single representative index per group is enough to pull the
        # matching mask/type_ids rows for that unique id sequence.
        representative = torch.zeros(num_unique, dtype=torch.long, device=flat_ids.device)
        representative.scatter_(0, inverse, torch.arange(n, device=flat_ids.device))

        unique_mask = flat_mask[representative]
        unique_type = flat_type[representative]

        unique_out = self._run_e0(unique_ids, unique_mask, unique_type)
        return unique_out[inverse]

    def _run_e0(self, ids: torch.Tensor, mask: torch.Tensor, type_ids: torch.Tensor) -> torch.Tensor:
        chunk_size = self.anchor_encode_chunk_size
        n = ids.size(0)
        if chunk_size is None or chunk_size >= n:
            return self._maybe_checkpoint(ids, mask, type_ids)

        outputs = [
            self._maybe_checkpoint(ids[start:start + chunk_size], mask[start:start + chunk_size],
                                   type_ids[start:start + chunk_size])
            for start in range(0, n, chunk_size)
        ]
        return torch.cat(outputs, dim=0)

    def _maybe_checkpoint(self, ids: torch.Tensor, mask: torch.Tensor, type_ids: torch.Tensor) -> torch.Tensor:
        if self.use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(self.e0, ids, mask, type_ids, use_reentrant=False)
        return self.e0(ids, mask, type_ids)

    def encode_entity(self, batch: dict) -> torch.Tensor:
        return self.e1(batch["input_ids"], batch["attention_mask"], batch["token_type_ids"])

"""
ARPM-KGC model (Sec 4.1-4.12), wiring every sub-module behind `ARPMConfig`
flags so any row of Table 3 (baseline / A1-A13 / Full) is reachable without
code changes.

Forward-pass ordering follows Algorithm 1 (Sec 4.9's page) rather than the
prose narrative order of Sections 4.4-4.10, which differ on one point (see
`hop_score_uses_prototype_query` in config.py): Algorithm 1 scores hops with
the raw query q (line 15, executed before q_p exists at line 19), while
Sec 4.6's prose says the hop scorer uses q_p. We default to the executable
Algorithm 1 reading and expose the prose reading as an opt-in flag.

Note on module instantiation vs. use: submodules whose role is entirely
config-gated (e.g. `prototype_generator`/`qp_interaction` when
`use_prototype_memory=False`, or `memory_gate` when
`use_adaptive_memory_gate=False`) are still constructed unconditionally in
`__init__` for simplicity, but may go uncalled on a given forward pass. This
is harmless for single-process/single-GPU training (unused parameters
simply receive no gradient); `Trainer` sets `find_unused_parameters=True`
when wrapping the model in DistributedDataParallel to keep this safe under
multi-GPU DDP as well. `structural_memory` and `memory_fusion` are the two
exceptions -- they are only constructed when their governing flag is on,
since a `None` submodule (rather than an unused live one) is the more
natural way to express "this branch does not exist" for the two ablations
(A9/A10) that explicitly test removing an entire memory source.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ARPMConfig
from .modules.encoders import DualTowerEncoder
from .modules.fusion import MemoryFusion, MemoryGate, QueryMemoryFusion, resolve_memory_gate
from .modules.prototype import PrototypeGenerator, QueryPrototypeInteraction, SlotGate, resolve_slot_gate
from .modules.retrieval import AnchorSelector
from .modules.structural import StructuralMemory


@dataclass
class ModelOutput:
    q: torch.Tensor                                # raw query repr, E0(h, r)
    qp: torch.Tensor                                # query after prototype interaction
    qe: torch.Tensor                                # final fused query repr
    tail_vector: torch.Tensor                       # E1(t)
    head_vector: torch.Tensor                       # E1(h)
    prototypes: Optional[torch.Tensor]              # [B, K, d] or None
    lambda_mem: torch.Tensor                        # [B]
    anchors: Optional[torch.Tensor]                 # [B, NA, d] or None
    anchor_mask: Optional[torch.Tensor]              # [B, NA] or None
    alpha: Optional[torch.Tensor]                    # [B, NA] or None
    raw_scores: Optional[torch.Tensor]                # [B, NA] or None
    beta: Optional[torch.Tensor]                      # [B, L] or None
    gamma: Optional[torch.Tensor]                      # [B, K] or None
    slot_mask: Optional[torch.Tensor]                   # [B, K_max] or None
    hop_has_data: Optional[torch.Tensor]                 # [B, L] or None


class ARPMKGCModel(nn.Module):
    def __init__(self, config: ARPMConfig, num_relations: int):
        super().__init__()
        self.config = config
        self.num_relations = num_relations
        self.gumbel_tau = config.gumbel_temperature_init  # updated once per epoch by the Trainer

        self.encoder = DualTowerEncoder(
            config.pretrained_model, config.pooling, config.dropout, config.tie_encoders,
        )
        d = self.encoder.hidden_size

        self.anchor_selector = AnchorSelector(config, d, num_relations)

        num_slots = config.max_prototype_slots if config.prototype_gating_mode == "gumbel_sigmoid_slots" \
            else config.num_prototypes
        self.prototype_generator = PrototypeGenerator(d, num_slots, config.weight_prototype_attention_by_alpha)
        self.slot_gate = SlotGate(d, num_relations, num_slots) \
            if config.prototype_gating_mode == "gumbel_sigmoid_slots" else None
        self.qp_interaction = QueryPrototypeInteraction(d)

        self.structural_memory = StructuralMemory(config, d, num_relations) \
            if config.use_structural_memory else None

        self.memory_fusion = MemoryFusion(d) \
            if (config.use_prototype_memory and config.use_structural_memory) else None
        self.query_memory_fusion = QueryMemoryFusion(d)
        self.memory_gate = MemoryGate(d, num_relations)

        self.log_inv_t = nn.Parameter(
            torch.tensor(1.0 / max(config.score_temperature, 1e-6)).log(),
            requires_grad=config.finetune_score_temperature,
        )

    def set_epoch(self, epoch: int) -> None:
        from .modules.gumbel import anneal_temperature
        self.gumbel_tau = anneal_temperature(
            epoch, self.config.gumbel_temperature_init,
            self.config.gumbel_temperature_min, self.config.gumbel_temperature_anneal_rate,
        )

    # ------------------------------------------------------------------
    def forward(self, batch: dict) -> ModelOutput:
        cfg = self.config
        q = self.encoder.encode_query(batch["query"])
        tail_vector = self.encoder.encode_entity(batch["tail_entity"])
        head_vector = self.encoder.encode_entity(batch["head_entity"])
        relation_ids = batch["relation_ids"]

        anchor_mask = batch["anchor_mask"]
        need_anchors = cfg.use_prototype_memory or cfg.use_structural_memory

        anchors = alpha = raw_scores = None
        if need_anchors:
            anchors = self.encoder.encode_anchors(
                batch["anchor_input_ids"], batch["anchor_attention_mask"], batch["anchor_token_type_ids"],
            )
            alpha, raw_scores, _gate = self.anchor_selector(
                q, anchors, anchor_mask, relation_ids, batch["anchor_hop"], gumbel_tau=self.gumbel_tau,
            )

        prototypes = gamma = slot_mask = None
        qp = q
        if cfg.use_prototype_memory and anchors is not None:
            prototypes, _rho = self.prototype_generator(q, anchors, alpha, anchor_mask)
            if cfg.prototype_gating_mode == "gumbel_sigmoid_slots":
                slot_mask, slot_weight, _zeta = resolve_slot_gate(
                    cfg, self.slot_gate, q, relation_ids, self.gumbel_tau, self.training,
                )
            else:
                slot_weight = None
            qp, m_p, gamma = self.qp_interaction(q, prototypes, slot_mask, slot_weight)
        else:
            m_p = None

        m_struct = beta = hop_has_data = None
        if cfg.use_structural_memory and anchors is not None:
            hop_query = qp if cfg.hop_score_uses_prototype_query else q
            m_struct, beta, hop_has_data = self.structural_memory(
                anchors, alpha, batch["anchor_hop"], batch["anchor_is_local"], relation_ids, hop_query,
                gumbel_tau=self.gumbel_tau,
            )

        m = self._fuse_memory(m_p, m_struct, q)
        qe = self.query_memory_fusion(qp, m)
        lambda_mem = resolve_memory_gate(cfg, self.memory_gate, q, relation_ids)

        return ModelOutput(
            q=q, qp=qp, qe=qe, tail_vector=tail_vector, head_vector=head_vector,
            prototypes=prototypes, lambda_mem=lambda_mem,
            anchors=anchors, anchor_mask=anchor_mask, alpha=alpha, raw_scores=raw_scores,
            beta=beta, gamma=gamma, slot_mask=slot_mask, hop_has_data=hop_has_data,
        )

    def _fuse_memory(self, m_p: Optional[torch.Tensor], m_struct: Optional[torch.Tensor],
                      q_like: torch.Tensor) -> torch.Tensor:
        if m_p is not None and m_struct is not None:
            return self.memory_fusion(m_p, m_struct)
        if m_p is not None:
            return m_p
        if m_struct is not None:
            return m_struct
        return torch.zeros_like(q_like)

    # ------------------------------------------------------------------
    def score(self, qe: torch.Tensor, prototypes: Optional[torch.Tensor], lambda_mem: torch.Tensor,
              candidate_vectors: torch.Tensor, apply_margin_diagonal: bool = False) -> Dict[str, torch.Tensor]:
        """S_q(t), S_p(t), and S(t|h,r) = S_q(t) + lambda_mem * S_p(t) (Sec 4.10).

        `candidate_vectors` is either the in-batch tail matrix [B, d]
        (training) or the full entity-embedding matrix [N, d] (evaluation).
        `apply_margin_diagonal` subtracts the additive margin from the
        diagonal (i.e. only valid when `candidate_vectors` is the in-batch
        tail matrix aligned index-for-index with the query batch).
        """
        inv_t = self.log_inv_t.exp()
        qe_n = F.normalize(qe, dim=-1)
        cand_n = F.normalize(candidate_vectors, dim=-1)

        cos_q = qe_n.mm(cand_n.t())  # [Bq, Nt]
        if apply_margin_diagonal:
            n = min(cos_q.size(0), cos_q.size(1))
            margin = torch.zeros_like(cos_q)
            idx = torch.arange(n, device=cos_q.device)
            margin[idx, idx] = self.config.additive_margin
            cos_q = cos_q - margin
        s_q = cos_q * inv_t

        s_p = None
        if prototypes is not None:
            proto_n = F.normalize(prototypes, dim=-1)  # [Bq, K, d]
            cos_p = torch.einsum("bkd,nd->bkn", proto_n, cand_n) * inv_t  # [Bq, K, Nt]
            tau_p = max(self.config.prototype_temperature, 1e-4)
            s_p = tau_p * torch.logsumexp(cos_p / tau_p, dim=1)  # [Bq, Nt]

        s_final = s_q if s_p is None else s_q + lambda_mem.unsqueeze(-1) * s_p
        return {"Sq": s_q, "Sp": s_p, "S": s_final}

    def self_negative_score(self, qe: torch.Tensor, head_vector: torch.Tensor) -> torch.Tensor:
        inv_t = self.log_inv_t.exp()
        qe_n = F.normalize(qe, dim=-1)
        head_n = F.normalize(head_vector, dim=-1)
        return (qe_n * head_n).sum(dim=-1) * inv_t

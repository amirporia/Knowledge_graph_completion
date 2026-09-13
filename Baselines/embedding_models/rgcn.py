from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import KGEModel, xavier_embedding

"""Verified against Schlichtkrull et al. 2018 (Modeling Relational Data with Graph
Convolutional Networks). Message-passing rule (Eq. 2), basis decomposition (Eq. 3),
and the link-prediction-specific degree normalization (see common/data.py's
build_message_passing_graph) all match the paper. Known, documented simplifications
relative to the paper's own link-prediction setup:
  - Only basis decomposition (Eq. 3) is implemented, not the block-diagonal
    alternative (Eq. 4) -- the paper reports block decomposition working best
    specifically on FB15k-237 ("two layers with block dimension 5x5"), so results on
    that dataset in particular may not match the paper's without it.
  - No edge dropout (the paper regularizes the encoder with dropout on edges before
    normalization: 0.2 for self-loops, 0.4 for others) or L2 penalty on the decoder
    (0.01 in the paper) -- both omitted here for now.
  - Trained through this repo's shared self-adversarial negative-sampling loss (see
    base_model.py) rather than the paper's own binary cross-entropy with uniform
    negative sampling (Eq. 7) -- the general one-shared-loss tradeoff explained in
    base_model.py's docstring, not specific to RGCN.
"""


class RGCNLayer(nn.Module):
    """Basis-decomposition relational graph convolution (Schlichtkrull et al. 2018),
    implemented in pure PyTorch (no torch_geometric dependency) so it runs anywhere
    the rest of this repo does.

    Messages are batched *per relation type* (a single (n_rel_edges, in) @ (in, out)
    matmul) rather than gathering a (E, in, out) weight tensor per edge — this is what
    keeps memory bounded on graphs with millions of edges (wiki5m).
    """

    def __init__(self, in_dim: int, out_dim: int, num_relations: int, num_bases: int):
        super().__init__()
        num_bases = min(num_bases, num_relations)
        self.num_relations = num_relations
        self.bases = nn.Parameter(torch.empty(num_bases, in_dim, out_dim))
        self.coeffs = nn.Parameter(torch.empty(num_relations, num_bases))
        self.self_loop_w = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.bases)
        nn.init.xavier_uniform_(self.coeffs)
        nn.init.xavier_uniform_(self.self_loop_w)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor,
                edge_norm: Optional[torch.Tensor]) -> torch.Tensor:
        out_dim = self.bases.size(-1)
        out = torch.zeros(x.size(0), out_dim, device=x.device, dtype=x.dtype)

        rel_weight = torch.einsum('rb,bio->rio', self.coeffs, self.bases)  # (num_rel, in, out)
        src, dst = edge_index[0], edge_index[1]

        for rel in range(self.num_relations):
            mask = edge_type == rel
            if not torch.any(mask):
                continue
            msg = x[src[mask]].mm(rel_weight[rel])                          # (n_edges_r, out)
            if edge_norm is not None:
                msg = msg * edge_norm[mask].unsqueeze(-1)
            out.index_add_(0, dst[mask], msg)

        out = out + x.mm(self.self_loop_w) + self.bias
        return out


class RGCN(KGEModel):
    """Schlichtkrull et al. 2018 — Modeling Relational Data with Graph Convolutional
    Networks. Two R-GCN layers encode entities from the training graph; a DistMult
    decoder scores triples on top of the resulting entity embeddings.

    Entity embeddings depend on the *whole graph*, not on the current mini-batch, so
    `encode_graph()` is called once per epoch by common/trainer.py (see
    `requires_graph_encode = True`) and cached in `self._entity_cache` for every batch/
    eval call within that epoch — this avoids re-running the graph convolution once per
    mini-batch, which is the main cost of training R-GCN naively.
    """

    distance_based = False
    requires_graph_encode = True

    def __init__(self, num_entities: int, num_relations: int, args):
        super().__init__(num_entities, num_relations, args)
        in_dim = getattr(args, 'rgcn_in_dim', args.embedding_dim)
        hidden_dim = getattr(args, 'rgcn_hidden_dim', args.embedding_dim)
        out_dim = args.embedding_dim
        num_bases = getattr(args, 'rgcn_num_bases', 30)
        dropout = getattr(args, 'dropout', 0.2)

        self.input_emb = xavier_embedding(num_entities, in_dim)
        self.layer1 = RGCNLayer(in_dim, hidden_dim, num_relations, num_bases)
        self.layer2 = RGCNLayer(hidden_dim, out_dim, num_relations, num_bases)
        self.dropout = nn.Dropout(dropout)
        self.rel_emb = xavier_embedding(num_relations, out_dim)

        self._entity_cache: Optional[torch.Tensor] = None

    def encode_graph(self, edge_index: torch.Tensor, edge_type: torch.Tensor,
                      edge_norm: torch.Tensor) -> None:
        x = self.input_emb
        x = torch.relu(self.layer1(x, edge_index, edge_type, edge_norm))
        x = self.dropout(x)
        x = self.layer2(x, edge_index, edge_type, edge_norm)
        self._entity_cache = x

    @property
    def entity_embedding(self) -> torch.Tensor:
        if self._entity_cache is None:
            raise RuntimeError('RGCN.encode_graph() must be called before scoring '
                                '(the trainer does this once per epoch automatically).')
        return self._entity_cache

    def score(self, h_idx, r_idx, t_idx):
        ent = self.entity_embedding
        h = ent[h_idx]
        r = self.rel_emb[r_idx]
        t = ent[t_idx]
        return torch.sum(h * r * t, dim=-1)

    def score_all(self, h_idx, r_idx):
        ent = self.entity_embedding
        query = ent[h_idx] * self.rel_emb[r_idx]
        return query.mm(ent.t())

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import KGEModel, xavier_embedding


class ConvE(KGEModel):
    """Dettmers et al. 2018 — Convolutional 2D Knowledge Graph Embeddings.
    (h, r) are reshaped into a 2D "image" and stacked; a conv + FC stack projects them
    back into embedding space, then that projection is scored against entity
    embeddings. Unlike the original paper's 1-N BCE training, here it plugs into the
    shared negative-sampling loss (see base_model.py) since only the (h, r) tower
    needs to run through the conv net either way — scoring one/many tails is just a
    dot product against the projected (h, r) vector.

    Verified against the paper: the architecture (stack -> BN -> dropout -> conv -> BN
    -> ReLU -> dropout -> flatten -> FC -> dropout -> BN -> ReLU -> dot with tail +
    bias, matching Eq. 1's f(vec(f([es;rr]*w))W)eo applying the nonlinearity f twice)
    and per-entity bias term match the paper/reference implementation exactly. Fixed
    one real gap: the paper independently tunes *three* dropout rates (embedding /
    feature-map / projection-layer dropout, its best setting being 0.2/0.2/0.3) — an
    earlier version of this file used one shared --dropout for all three, which is
    corrected below via three separate flags.
    """

    distance_based = False

    def __init__(self, num_entities: int, num_relations: int, args):
        super().__init__(num_entities, num_relations, args)
        dim = args.embedding_dim
        self.dim = dim
        self.emb_h, self.emb_w = getattr(args, 'conve_height', 10), dim // getattr(args, 'conve_height', 10)
        assert self.emb_h * self.emb_w == dim, 'embedding_dim must be divisible by conve_height'

        self.ent_emb = xavier_embedding(num_entities, dim)
        self.rel_emb = xavier_embedding(num_relations, dim)
        self.bias = nn.Parameter(torch.zeros(num_entities))

        num_filters = getattr(args, 'conve_filters', 32)
        kernel = getattr(args, 'conve_kernel', 3)
        # Paper's grid search tunes these three independently (best found: 0.2/0.2/0.3
        # on WN18/YAGO3-10/FB15k) rather than sharing one --dropout value.
        input_dropout = getattr(args, 'conve_input_dropout', 0.2)
        feature_dropout = getattr(args, 'conve_feature_dropout', 0.2)
        hidden_dropout = getattr(args, 'conve_hidden_dropout', 0.3)

        self.bn0 = nn.BatchNorm2d(1)
        self.conv = nn.Conv2d(1, num_filters, kernel_size=kernel, stride=1, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.input_drop = nn.Dropout(input_dropout)
        self.feat_drop = nn.Dropout(feature_dropout)
        conv_h = 2 * self.emb_h - kernel + 1
        conv_w = self.emb_w - kernel + 1
        self.fc = nn.Linear(num_filters * conv_h * conv_w, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.hidden_drop = nn.Dropout(hidden_dropout)

    def _project_hr(self, h_idx, r_idx):
        h = self.ent_emb[h_idx].view(-1, 1, self.emb_h, self.emb_w)
        r = self.rel_emb[r_idx].view(-1, 1, self.emb_h, self.emb_w)
        stacked = torch.cat([h, r], dim=2)                       # (B, 1, 2*emb_h, emb_w)
        x = self.bn0(stacked)
        x = self.input_drop(x)
        x = self.conv(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.feat_drop(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        return torch.relu(x)

    def score(self, h_idx, r_idx, t_idx):
        hr = self._project_hr(h_idx, r_idx)
        t = self.ent_emb[t_idx]
        return torch.sum(hr * t, dim=-1) + self.bias[t_idx]

    def score_all(self, h_idx, r_idx):
        hr = self._project_hr(h_idx, r_idx)
        return hr.mm(self.ent_emb.t()) + self.bias.unsqueeze(0)

    def forward(self, batch: dict) -> torch.Tensor:
        """Overrides the shared negative-sampling loss only to avoid redundant work:
        our negatives always corrupt the tail (neg_h == h, neg_r == r; see
        common/data.py::NegSamplingDataset), so the expensive conv+FC (h, r)
        projection only needs to run once per example instead of once for the
        positive and once per negative.
        """
        hr = self._project_hr(batch['h'], batch['r'])                          # (B, dim)
        pos_t = self.ent_emb[batch['t']]
        pos_score = torch.sum(hr * pos_t, dim=-1) + self.bias[batch['t']]      # (B,)

        neg_t = self.ent_emb[batch['neg_t']]                                    # (B, N, dim)
        neg_score = torch.einsum('bd,bnd->bn', hr, neg_t) + self.bias[batch['neg_t']]

        with torch.no_grad():
            neg_weight = F.softmax(neg_score * self.adv_temperature, dim=-1)
        pos_loss = -F.logsigmoid(pos_score).mean()
        neg_loss = -(neg_weight * F.logsigmoid(-neg_score)).sum(dim=-1).mean()
        return (pos_loss + neg_loss) / 2.0

"""StAR — Structure-Augmented Text Representation learning for efficient KGC
(Wang, Shen, Long, Zhou, Wang & Chang, WWW'21). https://doi.org/10.1145/3442381.3450043

Implements the paper's core model faithfully:
  - Section 3.1 Structure-Aware Triple Encoding: a *Siamese* (parameter-tied)
    Transformer encoder produces u = Pool(Enc([h;r])) and v = Pool(Enc([t]))
    (Eq. 4-7).
  - Section 3.2.1 Deterministic Representation Learning: c = [u; u*v; u-v; v]
    (Eq. 8) fed to an MLP binary classifier, trained with BCE (L^c, Eq. 12).
  - Section 3.2.2 Spatial Structure Learning: s^d = -||u - v||_2 (Eq. 11), trained
    with a margin hinge loss against sampled negatives (L^d, Eq. 13).
  - Section 3.3: total loss L = L^c + gamma * L^d (Eq. 14), gamma =
    args.structure_loss_weight.
  - Section 3.3.1: negatives are explicit, sampled per positive triple by
    corrupting head or tail uniformly at random (see star_data.py) — not
    in-batch contrastive negatives.

Deliberately not implemented (documented in config.py and the top-level README):
  - Section 3.4 self-adaptive ensemble with RotatE — a separate trained component.
  - Using s^c as the inference-time ranking basis over the full entity corpus — s^d
    is used instead, which is rank-equivalent to the shared evaluate.py's
    dot-product ranking for L2-normalized embeddings, and is very close in the
    paper's own ablation (Table 7: Hits@10 .701 vs .709 full model; MRR .406 vs
    .401 for s^d alone).
"""

from abc import ABC

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


def build_model(args) -> nn.Module:
    return StarBertModel(args)


class InteractionClassifier(nn.Module):
    """Eq. 8-9: c = [u; u*v; u-v; v] -> MLP -> single logit (equivalent to the
    paper's 2-way softmax's positive-class probability p2 via sigmoid)."""

    def __init__(self, hidden_size: int, mlp_hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size * 4, mlp_hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        c = torch.cat([u, u * v, u - v, v], dim=-1)
        return self.net(c).squeeze(-1)


class StarBertModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)

        # Siamese encoder (Section 3.1): hr_bert and tail_bert are the *same*
        # module (tied by reference, not deepcopy), matching "we keep the two
        # Transformer encoders ... parameter-tied for parameter efficiency" (Section 3.1).
        self.hr_bert = AutoModel.from_pretrained(args.pretrained_model)
        self.tail_bert = self.hr_bert

        self.classifier = InteractionClassifier(
            hidden_size=self.config.hidden_size,
            mlp_hidden=getattr(args, 'interaction_hidden_dim', 256),
            dropout=args.dropout,
        )

    def _encode(self, encoder, token_ids, mask, token_type_ids) -> torch.Tensor:
        outputs = encoder(input_ids=token_ids, attention_mask=mask,
                          token_type_ids=token_type_ids, return_dict=True)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return nn.functional.normalize(cls_output, dim=1)

    def encode(self, hr_token_ids, hr_mask, hr_token_type_ids,
               tail_token_ids, tail_mask, tail_token_type_ids):
        u = self._encode(self.hr_bert, hr_token_ids, hr_mask, hr_token_type_ids)
        v = self._encode(self.tail_bert, tail_token_ids, tail_mask, tail_token_type_ids)
        return u, v

    def forward(self, hr_token_ids, hr_mask, hr_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                only_ent_embedding: bool = False, **kwargs) -> dict:
        if only_ent_embedding:
            return self.predict_ent_embedding(tail_token_ids, tail_mask, tail_token_type_ids)

        u, v = self.encode(hr_token_ids, hr_mask, hr_token_type_ids,
                           tail_token_ids, tail_mask, tail_token_type_ids)
        # Keep the 'hr_vector' / 'tail_vector' key names so predict.py / evaluate.py
        # (unchanged from SimKGC) work without modification.
        return {'hr_vector': u, 'tail_vector': v}

    def interaction_logits(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.classifier(u, v)

    @torch.no_grad()
    def predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids, **kwargs) -> dict:
        ent_vectors = self._encode(self.tail_bert, tail_token_ids, tail_mask, tail_token_type_ids)
        return {'ent_vectors': ent_vectors.detach()}

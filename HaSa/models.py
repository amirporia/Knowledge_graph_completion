"""HaSa — Hardness and Structure-Aware Contrastive Knowledge Graph Embedding
(Zhang, Zhang & Molybog, WWW'24). https://doi.org/10.1145/3589334.3645564

Faithful to the paper's core design:
  - A *single* shared text encoder f(.) (Algorithm 1: "current encoder f") is applied
    separately to head text, relation text, and tail text -- unlike SimKGC/StAR in
    this repo, HaSa does not concatenate head+relation into one input string. f(.) =
    pretrained LM -> [CLS] pooling -> Linear -> LayerNorm -> Dropout, reducing to
    args.embedding_dim (Section 7.1: "an additional neural network (a linear layer, a
    Layer normalization, and a dropout layer ...) to reduce the embedding dimension").
  - An aggregation function g(e_h, e_r) combining head and relation embeddings into
    the query embedding e_hr. The paper specifies g as a GRU ("we follow the setting
    of InfoNCE [25] to use the gated recurrent unit (GRU) neural network to model the
    aggregation function g(.,.)", Section 3): e_h and e_r are fed as a length-2
    sequence, and the GRU's final hidden state is e_hr.
  - The hardness- and structure-aware loss (Eq. 6-9, Algorithm 1) is computed in
    trainer.py (`_hasa_loss` / `_false_negative_term`), since it operates across the
    whole training batch plus extra structure-sampled candidates, not on a single
    forward call.

One deliberate addition beyond the paper: e_h/e_r/e_t/e_hr are L2-normalized (the
paper's equations use raw dot products, but normalizing is standard practice for
contrastive-loss stability, and is what SimKGC/StAR already do in this repo). Ranking
under the shared, unmodified evaluate.py's dot-product scoring — which matches the
paper's own exp(e_hr^T e_t) — is unaffected by this choice since it's applied
uniformly to the query and every candidate. HaSa+'s (Section 6) extra "negative
query" loss term is not implemented; flag if you want it added.

BUGFIX (multi-GPU / DistributedDataParallel correctness, log_inv_t): `log_inv_t`
used to be read only in trainer.py::_hasa_loss, via
`get_model_obj(self.model).log_inv_t` -- entirely outside `forward()`. Under
DistributedDataParallel, a parameter's gradient is only reliably synchronized
across GPUs if it's reachable from the tensors `forward()` itself returns, at the
moment `forward()` returns (DDP walks that graph right then to know which
parameters to expect a gradient for). `log_inv_t` is a bare `nn.Parameter`, never
combined with anything inside `forward()`, so DDP would treat it as "unused" every
iteration and stop synchronizing it across replicas -- each GPU's copy would then
silently drift to a different temperature over the course of multi-GPU training,
with no error raised. `forward()` now returns `log_inv_t` directly alongside
`hr_vector`/`tail_vector`/`head_vector`, making it part of the same tracked graph;
trainer.py reads it from `outputs['log_inv_t']` instead of the module directly.
(This is exactly the same class of bug as StAR's `interaction_logits`, see
StAR/models.py's module docstring for a fuller explanation of the underlying DDP
mechanism.)

BUGFIX (multi-GPU / DistributedDataParallel correctness, false-negative
candidates): trainer.py::_hasa_loss used to encode the structure-sampled
false-negative candidates via a raw `get_model_obj(self.model).encode_text(...)`
call -- i.e. directly on the *unwrapped* module, outside of and in addition to the
one `self.model(**batch_dict)` call per iteration that DDP actually wraps and
hooks. That created a second autograd path through `encoder`/`proj` that DDP's
reducer never registered when it set up its per-iteration "expect one gradient per
parameter" bookkeeping during the single tracked forward() call. When
`loss.backward()` then walked the merged graph (main batch + candidates), DDP's
mark-ready accounting desynced and raised:
    RuntimeError: Expected to mark a variable ready only once ...
    Parameter at index 0 with name .log_inv_t has been marked as ready twice.
(log_inv_t is just the parameter whose gradient happens to close out the graph
first; the underlying cause is the untracked second call through encoder/proj,
not log_inv_t itself.) `_set_static_graph()` is not a safe workaround here either,
since some batches have zero false-negative candidates -- a genuinely
varying graph shape from iteration to iteration.

Fix: candidate tail tokens are now passed into this single `forward()` call
(`cand_tail_token_ids` / `cand_tail_mask` / `cand_tail_token_type_ids`) and encoded
with the same `encode_text()` call used for everything else, so every use of
`encoder`/`proj`/`log_inv_t` happens inside the one DDP-tracked forward() per
iteration. `outputs['cand_vector']` is returned (None when no candidates exist
for this batch); trainer.py's `_hasa_loss` now just consumes it instead of
encoding it itself.
"""

from abc import ABC

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


def build_model(args) -> nn.Module:
    return HaSaBertModel(args)


class HaSaBertModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)

        # Single shared encoder f(.), applied to head / relation / tail text alike
        # (and to structure-sampled false-negative candidates, encoded within this
        # same forward() call -- see the module docstring's DDP bugfix note).
        self.encoder = AutoModel.from_pretrained(args.pretrained_model)

        self.proj = nn.Sequential(
            nn.Linear(self.config.hidden_size, args.embedding_dim),
            nn.LayerNorm(args.embedding_dim),
            nn.Dropout(args.dropout),
        )

        # g(.,.): GRU aggregation of [e_h, e_r] -> e_hr (Section 3).
        self.aggregator = nn.GRU(input_size=args.embedding_dim,
                                 hidden_size=args.embedding_dim,
                                 batch_first=True)

        # Practical addition beyond the paper: a learnable inverse temperature applied
        # to similarity scores before exp(.) in the Eq. 6-9 loss (see trainer.py). The
        # paper's equations exponentiate a raw dot product of L2-normalized vectors,
        # which is bounded to [-1, 1] and gives a very flat, hard-to-optimize softmax;
        # every contrastive KGE method HaSa itself compares against (RotatE-style
        # self-adversarial sampling, SimKGC) uses a temperature for exactly this
        # reason. Initialized from args.t like SimKGC's log_inv_t.
        self.log_inv_t = nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=True)

    def encode_text(self, token_ids: torch.Tensor, mask: torch.Tensor,
                    token_type_ids: torch.Tensor) -> torch.Tensor:
        """f(x): pretrained encoder -> [CLS] -> Linear/LayerNorm/Dropout -> L2-normalize."""
        outputs = self.encoder(input_ids=token_ids, attention_mask=mask,
                               token_type_ids=token_type_ids, return_dict=True)
        cls_output = outputs.last_hidden_state[:, 0, :]
        projected = self.proj(cls_output)
        return nn.functional.normalize(projected, dim=1)

    def aggregate(self, e_h: torch.Tensor, e_r: torch.Tensor) -> torch.Tensor:
        """g(e_h, e_r) via a GRU over the 2-step sequence [e_h, e_r] (Section 3)."""
        seq = torch.stack([e_h, e_r], dim=1)            # (B, 2, dim)
        _, h_n = self.aggregator(seq)                    # h_n: (1, B, dim)
        return nn.functional.normalize(h_n.squeeze(0), dim=1)

    def forward(self, head_token_ids, head_mask, head_token_type_ids,
                relation_token_ids, relation_mask, relation_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                cand_tail_token_ids=None, cand_tail_mask=None, cand_tail_token_type_ids=None,
                only_ent_embedding: bool = False, **kwargs) -> dict:
        if only_ent_embedding:
            return self.predict_ent_embedding(tail_token_ids, tail_mask, tail_token_type_ids)

        e_h = self.encode_text(head_token_ids, head_mask, head_token_type_ids)
        e_r = self.encode_text(relation_token_ids, relation_mask, relation_token_type_ids)
        e_t = self.encode_text(tail_token_ids, tail_mask, tail_token_type_ids)
        e_hr = self.aggregate(e_h, e_r)

        # BUGFIX (DDP "Expected to mark a variable ready only once"): encode the
        # structure-sampled false-negative candidates (Eq. 8-13) here, inside this
        # single tracked forward() call, instead of via a second, untracked call to
        # encode_text() from trainer.py after this call returns. See the module
        # docstring for the full explanation. `cand_tail_token_ids` is None whenever
        # a batch has no false-negative candidates to sample (e.g. isolated head
        # entities with no <=2-hop neighbours).
        cand_vector = None
        if cand_tail_token_ids is not None:
            cand_vector = self.encode_text(cand_tail_token_ids, cand_tail_mask, cand_tail_token_type_ids)

        # Key names kept as 'hr_vector' / 'tail_vector' so the shared, unmodified
        # predict.py / evaluate.py (dot-product filtered-MRR ranking, matching the
        # paper's own exp(e_hr^T e_t) scoring) work without changes.
        # 'log_inv_t' is returned here (rather than read directly off the module
        # later) purely for DDP correctness -- see the module docstring's BUGFIX note.
        return {'hr_vector': e_hr, 'tail_vector': e_t, 'head_vector': e_h,
               'log_inv_t': self.log_inv_t, 'cand_vector': cand_vector}

    @torch.no_grad()
    def predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids, **kwargs) -> dict:
        ent_vectors = self.encode_text(tail_token_ids, tail_mask, tail_token_type_ids)
        return {'ent_vectors': ent_vectors.detach()}

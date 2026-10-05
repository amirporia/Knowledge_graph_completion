# ARPM-KGC: Adaptive Relation-Aware Prototype Memory for Knowledge Graph Completion

A reference implementation of **ARPM-KGC**, a text-based knowledge graph completion (KGC) model that extends relation-aware anchor retrieval with query-conditioned anchor selection, multi-prototype semantic memory, adaptive structural (hop-distance) memory, and an adaptive gate that controls how strongly each memory source influences the final prediction.

The method builds on the relation-aware anchor enhancement framework of Yuan et al. (2025) [RAA-KGC] and is evaluated under the standard filtered-ranking KGC protocol on WN18RR, FB15k-237, and Wikidata5M.

## Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Repository Structure](#repository-structure)
- [Data Preparation](#data-preparation)
- [Proposal-to-Code Mapping](#proposal-to-code-mapping)
- [Usage](#usage)
- [Hyperparameters](#hyperparameters)
- [Ablation Study](#ablation-study)
- [Checkpoint Selection](#checkpoint-selection)
- [Design Decisions and Implementation Notes](#design-decisions-and-implementation-notes)
- [Validation Performed](#validation-performed)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Overview

Given a query $(h, r, ?)$, ARPM-KGC retrieves a bounded pool of relation-aware anchor triples from the training graph, both from the local neighborhood of $h$ (at graph distances $1, \dots, N$) and from the global set of relation-$r$ triples. It then makes four query-conditioned decisions over that pool:

1. **Anchor selection (RQ1).** Which anchors are relevant to this query, via query-conditioned attention weights $\alpha_i$ rather than uniform or random sampling.
2. **Prototype organization (RQ2).** Which distinct semantic modes the selected anchors represent, via $K$ learned prototype vectors instead of a single averaged memory vector.
3. **Structural aggregation (RQ3).** Which graph-hop distances provide useful evidence, via a learned, hop-availability-masked weighting over per-hop structural memories.
4. **Memory gating (RQ4).** How strongly semantic and structural memory should influence the final score, via two independent, query-conditioned gates $\lambda_p, \lambda_s$.

The final score is additive and does not modify the query representation:

$$
S(t \mid h, r) = S_q(t) + \lambda_p S_p(t) + \lambda_s S_{\text{struct}}(t)
$$

so a query for which memory is unhelpful degrades gracefully to the underlying query–entity similarity, with both gates independently discardable.

## Requirements

- Python 3.8+
- PyTorch (tested under 1.12+ and 2.x)
- `transformers` (`AutoModel`, `AutoConfig`, `AutoTokenizer`, and the linear/cosine warmup schedulers)
- `tqdm`

No specific package versions are pinned in this repository; any reasonably recent combination of the above is expected to work. GPU training additionally requires CUDA and, for multi-GPU runs, `torch.distributed`/`torchrun`.

## Repository Structure

```
ARPM_KGC/
  setting/      config.py, logger_config.py
  utils/        triplet.py, dict_hub.py, candidate_pool.py, doc.py, triplet_mask.py, utils.py
  model/        modules.py, models.py, trainer.py
  evaluation/   metric.py, predict.py, evaluate.py, eval_wiki5m_trans.py
  preprocess/   preprocess.py
  main.py
```

## Data Preparation

From the directory that *contains* `ARPM_KGC/`, run:

```bash
python -m ARPM_KGC.preprocess.preprocess --task wn18rr
```

This produces the tokenized triple files and `entities.json` under `data/<TASK>/` expected by `utils/dict_hub.py` and `utils/doc.py`. The same command works for `fb15k237`, `wiki5m_trans`, and `wiki5m_ind`.

## Proposal-to-Code Mapping

| Proposal section | Defines | Implementation |
| --- | --- | --- |
| Relation-Aware Candidate Memory | $\mathcal{A}_{\text{local}}(h,r)$, $\mathcal{A}_{\text{global}}(r)$, budget $M$ | `utils/candidate_pool.py::CandidatePoolBuilder`, `utils/triplet.py::LinkGraph.get_hop_layers` |
| Query-Conditioned Anchor Selection (RQ1) | $\alpha_i$, weighted anchor set $\mathcal{W}_q$ | `model/models.py::ARPMModel._build_memory` |
| Diversity Regularization | $\mathcal{L}_{\text{div}}$ | `model/modules.py::diversity_loss` |
| Multi-Prototype Semantic Memory (RQ2) | `ProtoGen`, prototypes $p_k$ | `model/modules.py::ProtoGen` |
| Adaptive Structural Memory (RQ3) | $m^{(\ell)}$, $G_{\text{hop}}$, $\beta_\ell$, $m_{\text{struct}}$, empty-hop safeguards | `model/models.py::ARPMModel._structural_memory`, `model/modules.py::HopScorer` |
| Adaptive Memory-Aware Tail Prediction (RQ4) | $S_q, S_p, S_{\text{struct}}$, $G_\lambda$, $S(t\mid h,r)$ | `model/models.py::ARPMModel.score_query/score_prototypes/score_struct/combined_score`, `model/modules.py::MemoryGate` |
| Training Objective, plus $\mathcal{L}_{\text{combined}}$ (added here; see [Design Decisions](#training-the-memory-gate)) | $\mathcal{L} = \mathcal{L}_{\text{query}} + \eta_p \mathcal{L}_{\text{proto}} + \eta_s \mathcal{L}_{\text{struct}} + \eta_{\text{div}} \mathcal{L}_{\text{div}} + \eta_c \mathcal{L}_{\text{combined}}$ | `model/trainer.py::Trainer._compute_losses` |
| Discrete anchor gating (A11) | $c_i^{\text{ST}}$ | `model/modules.py::gumbel_sigmoid_gate` |
| Discrete hop selection (A12) | $\beta_\ell^{\text{ST}}$, top-$k$ | `model/modules.py::gumbel_softmax_topk` |
| Discrete prototype gating (A13) | $\omega_k^{\text{ST}}$ | `model/modules.py::gumbel_sigmoid_slot_gate`, `PrototypeActivationScorer` |
| Memory Gate Analysis | Per-query $\lambda_p, \lambda_s$ | `evaluation/evaluate.py::_save_prediction_details` (written to `task_hrt_*.json`) |

`Example.vectorize(test=True)` produces the query encoding $E_0(h, r)$; `vectorize(test=False)` produces the anchor encoding $E_0(h_i, r, t_i)$. Both are reused unchanged from the underlying dual-encoder pipeline; only *which* triplets are vectorized as anchors changes.

## Usage

```bash
# Train (core model, all defaults from the proposal)
python -m ARPM_KGC.main --task wn18rr

# Evaluate the best checkpoint (filtered ranking: MRR, Hits@1/3/10/50)
python -m ARPM_KGC.evaluation.evaluate --task wn18rr --is-test

# Large-scale wiki5m_trans (sharded entity embeddings)
python -m ARPM_KGC.evaluation.eval_wiki5m_trans --task wiki5m_trans --is-test

# Multi-GPU training via torchrun
torchrun --nproc_per_node=2 -m ARPM_KGC.main --task wn18rr
```

All standard training flags (`--batch-size`, `--lr`, `--epochs`, `--pretrained-model`, `--pooling`, `--use-amp`, ...) are supported identically across single- and multi-GPU runs.

## Hyperparameters

| Flag | Meaning | Default |
| --- | --- |---------|
| `--num-hops` | $N$: max graph distance beyond hop-0, yielding $N+1$ local categories (hop-0 .. hop-$N$) | 2       |
| `--num-prototypes` | $K$: semantic prototypes, one of $\{1,2,4,8\}$ | 4       |
| `--local-per-hop-budget` | Max local anchors sampled per hop (including hop-0) | 5       |
| `--global-budget` | Max global anchors sampled from $\mathcal{T}_r$ | 10      |
| `--anchor-budget` | $M$: total candidate-pool cap after merging local and global | 20      |
| `--retrieval-temperature` | $\tau_r$: softmax temperature for anchor weights $\alpha_i$ | 0.1     |
| `--proto-temperature` | $\tau_p$: log-sum-exp pooling temperature for $S_p(t)$ | 0.1     |
| `--eta-proto` | $\eta_p$: weight of $\mathcal{L}_{\text{proto}}$ | 0.1     |
| `--eta-struct` | $\eta_s$: weight of $\mathcal{L}_{\text{struct}}$ | 0.1     |
| `--eta-div` | $\eta_{\text{div}}$: weight of $\mathcal{L}_{\text{div}}$ | 0.01    |
| `--eta-combined` | $\eta_c$: weight of $\mathcal{L}_{\text{combined}}$, the sole loss term that trains $G_\lambda$ (see [Design Decisions](#training-the-memory-gate)) | 0.1     |
| `--disable-link-graph` | Disable local candidates entirely (equivalent to ablation A7) | off     |
| `--checkpoint-metric` | Metric used for best-checkpoint selection ([details](#checkpoint-selection)) | `mrr`   |
| `--full-eval-every-n-epochs` | Cadence of the full filtered-ranking validation pass | 1       |
| `--full-eval-batch-size` | Batch size for the full validation pass | 8       |

## Ablation Study

Every row of the ablation table is reachable with a single flag, no code changes required.

| Ablation | Description | Command |
| --- | --- | --- |
| A1  | Random anchors instead of query-conditioned selection | `--random-anchor-selection` |
| A2  | No diversity regularization | `--eta-div 0` |
| A3  | Single prototype | `--num-prototypes 1` |
| A4  | Fixed vs. adaptive prototype count | compare default (fixed $K$) against A13 |
| A5  | Fixed/uniform hop weighting | `--uniform-hop-weighting` |
| A6  | No global relation memory | `--global-budget 0` |
| A7  | No local structural memory | `--disable-link-graph` |
| A8  | Fixed (non-adaptive) $\lambda_p, \lambda_s$ | `--fixed-lambda-p 0.5 --fixed-lambda-s 0.5` |
| A9  | No $S_p$ term | `--fixed-lambda-p 0` |
| A10 | No $S_{\text{struct}}$ term | `--fixed-lambda-s 0` |
| A11 | Gumbel-Sigmoid anchor gating | `--use-gumbel-anchor` |
| A12 | Gumbel-Softmax hop selection | `--use-gumbel-hop` |
| A13 | Gumbel-Sigmoid prototype slot gating | `--use-gumbel-proto` |
| Full | All core components | (defaults) |

```bash
python -m ARPM_KGC.main --task wn18rr --random-anchor-selection        # A1
python -m ARPM_KGC.main --task wn18rr --eta-div 0                      # A2
python -m ARPM_KGC.main --task wn18rr --num-prototypes 1               # A3
python -m ARPM_KGC.main --task wn18rr --uniform-hop-weighting          # A5
python -m ARPM_KGC.main --task wn18rr --global-budget 0                # A6
python -m ARPM_KGC.main --task wn18rr --disable-link-graph             # A7
python -m ARPM_KGC.main --task wn18rr --fixed-lambda-p 0.5 --fixed-lambda-s 0.5  # A8
python -m ARPM_KGC.main --task wn18rr --fixed-lambda-p 0               # A9
python -m ARPM_KGC.main --task wn18rr --fixed-lambda-s 0               # A10
python -m ARPM_KGC.main --task wn18rr --use-gumbel-anchor              # A11
python -m ARPM_KGC.main --task wn18rr --use-gumbel-hop                 # A12
python -m ARPM_KGC.main --task wn18rr --use-gumbel-proto               # A13
python -m ARPM_KGC.main --task wn18rr                                  # Full
```

An additional diagnostic control, not part of the proposal's ablation table, freezes the memory gate at random initialization:

```bash
python -m ARPM_KGC.main --task wn18rr --eta-combined 0
```

This is useful for confirming that $\mathcal{L}_{\text{combined}}$ is necessary, not merely sufficient, for $\lambda_p, \lambda_s$ to move from their initial values (see [Validation Performed](#validation-performed)); it is not the intended operating point for any reported result, since ablation A8 and the Memory Gate Analysis both presuppose a trained gate.

## Checkpoint Selection

Best-checkpoint selection uses the **same filtered-ranking protocol as final evaluation** — forward and backward MRR/Hits@1/3/10/50 against the full entity dictionary (`evaluation/evaluate.py::evaluate_predictor`, wrapped around the in-memory model via `ARPMPredictor.from_model`) — rather than a cheap in-batch proxy.

This distinction matters because in-batch accuracy (distinguishing the correct tail from the other tails present in the same mini-batch) and filtered ranking (distinguishing the correct tail from the entire entity dictionary, with known correct alternatives masked out) are tasks of different difficulty; a checkpoint selected on the former can be a poor choice under the latter.

- `Trainer.eval_epoch` runs every epoch and logs `Acc@1`/`Acc@3`/`loss` computed in-batch. This is a cheap training-health signal, explicitly logged as **not used for checkpoint selection**, and never updates `best_metric`.
- `Trainer._compute_full_validation_metrics` runs the real protocol at epoch boundaries: it wraps the current in-memory model as a predictor, encodes every entity, and calls the same `evaluate_predictor` used by the standalone `evaluation/evaluate.py` script (with `save_details=False` to avoid writing per-query prediction files every epoch). Its output — `mean_rank`, `mrr`, `hit@1/3/10/50` — determines `is_best` via `--checkpoint-metric` (default `mrr`).
- Because encoding the full entity dictionary every epoch is expensive for large graphs, this pass only runs at epoch boundaries and only every `--full-eval-every-n-epochs` epochs. Skipped epochs still checkpoint `model_last.mdl` for resuming, but never overwrite `model_best.mdl`.

`evaluation/predict.py::ARPMPredictor.from_model` and `evaluation/evaluate.py::evaluate_predictor`/`save_details` are shared between the trainer and the standalone evaluation scripts, so there is exactly one code path computing MRR/Hits@k.

## Design Decisions and Implementation Notes

The proposal specifies the model's math precisely but is necessarily silent on some engineering choices. This section documents the resulting decisions so results remain interpretable and reproducible.

### Training the memory gate

The proposal's training objective, $\mathcal{L} = \mathcal{L}_{\text{query}} + \eta_p \mathcal{L}_{\text{proto}} + \eta_s \mathcal{L}_{\text{struct}} + \eta_{\text{div}} \mathcal{L}_{\text{div}}$, is defined entirely in terms of $S_q$, $S_p$, $S_{\text{struct}}$, and the anchor weights $\alpha_i$; none of its four terms is a function of $\lambda_p$ or $\lambda_s$. The only quantity that depends on the memory gate $G_\lambda$ is the combined score $S(t\mid h,r) = S_q(t) + \lambda_p S_p(t) + \lambda_s S_{\text{struct}}(t)$, and the proposal uses that quantity only for ranking, at inference and as an in-batch diagnostic — not in the loss. Taken literally, this leaves $G_\lambda$ at its random initialization for the entire training run, which is inconsistent with treating $\lambda_p, \lambda_s$ as **learned** gates (the proposal's own framing), with ablation A8's premise that fixed gates should be compared against learned ones, and with the Memory Gate Analysis, all of which assume the gates carry a trained signal.

This implementation closes the gap with an auxiliary cross-entropy loss on the combined score:

$$
\mathcal{L}_{\text{combined}} = \text{CE}\big(S(\cdot \mid h, r), y\big)
$$

weighted by `--eta-combined` (default 0.1). This is the only term in `Trainer._compute_losses` whose gradient reaches `MemoryGate`. It does not alter the definition of $S(t\mid h,r)$ used for ranking anywhere in the pipeline — it exists solely to give $G_\lambda$ a training signal — and is an addition beyond the proposal's stated objective rather than a literal reading of it.

This also resolves a `DistributedDataParallel` failure under multi-GPU training (`RuntimeError: Expected to have finished reduction in the prior iteration...`), since DDP requires every trainable parameter to participate in the backward pass each iteration unless told otherwise, and `MemoryGate`'s two parameter tensors were previously the only ones that never did. `Trainer._setup_device` accordingly passes `find_unused_parameters=True` to `DistributedDataParallel`. This remains necessary even with $\mathcal{L}_{\text{combined}}$ enabled, because the `--fixed-lambda-p`/`--fixed-lambda-s` ablations (A8–A10) still bypass `MemoryGate`'s output in the forward pass via `torch.full_like`, so the overridden channel's absence from the forward computation must be declared to DDP's static-graph reduction on every run using those flags.

`--eta-combined 0` reproduces the gate-frozen-at-initialization behavior of a literal reading of the proposal's objective. It is retained as a diagnostic control, not the intended default; any result reported under the "learned gate" condition (ablation A8's baseline, or "Full") should use `--eta-combined > 0`.

### Temperature scaling is training-only

$S_q$, $S_p$, $S_{\text{struct}}$, and the combined $S(t\mid h,r)$ returned by `model/models.py` and used for ranking (evaluation, the Memory Gate Analysis) are always the raw, paper-exact similarities, with no temperature or margin applied. `model/trainer.py` applies a shared `exp(log_inv_t)` inverse temperature — and, for $S_q$ only, the additive margin and self-negative term — solely to sharpen the in-batch cross-entropy losses, including $\mathcal{L}_{\text{combined}}$. Since this is a positive monotonic rescaling applied identically inside each loss, it affects optimization dynamics but never the ranking any of the four scores would produce.

### False-negative masking

The shared `triplet_mask` filtering in-batch false negatives is applied identically to $S_q$, $S_p$, $S_{\text{struct}}$, and the combined score; an unmasked false negative would otherwise corrupt all four losses, not only the query loss. Pre-batch negatives are not implemented.

### Zero-candidate queries

If `CandidatePoolBuilder.build(...)` returns an empty list — an isolated head entity, or a relation seen nowhere else in training — the model does not fall back to the query's own true triple as an anchor, since that would leak the label. Instead, every candidate slot is the shared padding placeholder, `candidate_valid_mask` is all-`False` for that query, $\alpha_i$ and $\mathcal{L}_{\text{div}}$ are exactly zero, and prototypes/$m_{\text{struct}}$ are zero vectors (the structural-memory constant $\epsilon$ exists precisely for this case). This path was exercised directly via `--anchor-budget 0` and produces finite, non-NaN losses, including $\mathcal{L}_{\text{combined}}$, since $S_p$ and $S_{\text{struct}}$ degrade gracefully to zero-vector similarities rather than NaN.

### Link-graph ablation semantics

`--disable-link-graph` removes local anchors from the entire pipeline, not only from structural memory — prototype construction (which draws from the same weighted anchor set $\mathcal{W}_q$) loses local evidence as well. Under the proposal's own definition of $m_{\text{struct}}$ as built exclusively from local anchors, A7 ("no local structural memory") and A10 ("drop the $S_{\text{struct}}$ term") are the same removal in principle. This implementation keeps them distinct and both runnable: A10 (`--fixed-lambda-s 0`) drops $S_{\text{struct}}$ from the final score while local anchors still inform the prototypes; A7 (`--disable-link-graph`) removes local anchors from the pipeline entirely. Use whichever isolates the quantity under study.

### Adaptive prototype count

The proposal specifies a *fixed* $K \in \{1,2,4,8\}$ for the core model and introduces adaptive $K_q$ only as the optional Gumbel extension A13. Ablation A4 ("fixed prototype count instead of adaptive $K$") is accordingly read as the comparison between Full (fixed $K$, the default) and A13 (`--use-gumbel-proto`, adaptive $K_q$), rather than as an independent standalone flag.

### Scoring function choices

The proposal leaves $S_P(a_i, q, k)$ (prototype attention) and the exact form of $G_{\text{hop}}(q, \ell)$/$G_\lambda(q)$ unspecified beyond naming their roles. `model/modules.py::ProtoGen` implements $S_P$ as a per-prototype bilinear form $u_{ik} = a_i^\top W_k q / \sqrt{d}$, the natural generalization of dot-product attention to $K$ independent slots. `HopScorer` and `MemoryGate` are each implemented as a single linear layer whose $\ell$-th/$k$-th output channel is the corresponding score — the simplest faithful realization of "a function of $q$ and a discrete index."

Candidate retrieval reads exclusively from the training graph (`utils/dict_hub.py::get_train_triplet_dict`, independent of `args.is_test`), so no validation or test label ever enters the candidate pool, at either training or evaluation time. The candidate pipeline — retrieval scoring, diversity, prototype attention, and hop-specific aggregation — is vectorized over a padded `(batch, max_candidates)` grid with a single batched encoder call per pass.

### Hop indexing

Local hop 0 is defined as "same head and relation" (graph distance 0, no traversal), not "nearest graph neighbor," and `--num-hops N` yields $N+1$ local categories, not $N$: hop-0 (other valid tails for the exact same $(h,r)$ pair), hop-1 (graph distance 1), ..., hop-$N$ (graph distance $N$). `--num-hops 2` therefore allocates three structural-memory slots (`ARPMModel.num_hop_slots`). Global candidates and padding slots are tagged `NO_HOP = -1` specifically so they cannot be confused with a genuine local hop-0 candidate. `_global_candidates` also excludes any pair sharing the query's head — not only its exact true tail — since head-sharing pairs are already covered by the hop-0 category, keeping "global" strictly meaning "a different head."

### Empty-hop and no-local-anchor safeguards

`_structural_memory` implements two parameter-free safeguards:

1. **Some hops empty, not all.** Hop-selection weighting — the dense softmax, its `--use-gumbel-hop` (A12) counterpart, and the `--uniform-hop-weighting` (A5) ablation — is masked so that a hop with zero local anchors receives exactly zero weight, rather than relying on `G_hop` to learn this from data. Verified adversarially: forcing `G_hop` to output a large logit for a deliberately empty hop still yields exactly zero weight for it after masking.
2. **No local anchor at any hop.** `m_struct` is hard-set to the zero vector via `torch.where` (an element-wise replacement rather than a multiplicative mask, which would let a NaN from a degenerate all-hops-masked softmax propagate as `0 * NaN = NaN`), and `forward()` hard-overrides `lambda_s <- 0` for that query, after any `--fixed-lambda-s` ablation override. This override is differentiable with respect to the surviving branch, so $\mathcal{L}_{\text{combined}}$ continues to train `G_lambda` normally on queries where the override does not fire.

Implementing safeguard 2 surfaced a related off-by-one in `CandidatePoolBuilder._local_candidates`/`_same_head_candidates`: the per-hop budget was previously checked after appending a candidate, so `--local-per-hop-budget 0` still admitted one candidate per non-empty hop. The check now runs before appending, so a budget of 0 yields exactly 0 local candidates. `--num-hops 0` is consequently a valid configuration — hop-0 only, with no graph traversal — distinct from and more surgical than `--disable-link-graph` (A7), which removes every local category including hop-0.

### Known limitation: candidate-pool union

$\mathcal{A}(h,r) = \mathcal{A}_{\text{local}}(h,r) \cup \mathcal{A}_{\text{global}}(r)$ is currently implemented as a concatenation rather than a true set union. `_global_candidates` deduplicates against hop-0 by excluding any pair sharing the query's head, but a triple reached via graph traversal at hop $\ell \geq 1$ can still be independently sampled into the global pool, appearing twice under two different tags. This has no correctness impact — the local copy still feeds structural memory correctly, and the extra global copy only adds marginal retrieval/prototype weight to that anchor — but affects per-query candidate-pool inspection. A full fix would track seen $(h_i, t_i)$ pairs per query across both pools.

## Validation Performed

The full pipeline — candidate retrieval, anchor selection, diversity loss, prototype construction, structural memory, gating, combined score, $\mathcal{L}_{\text{combined}}$, training loss, checkpointing, and filtered-ranking evaluation — was exercised end-to-end on a small synthetic dataset with a randomly initialized, fully local (no network access) tiny BERT encoder. This covered:

- A full `train_loop` with checkpoint save/load and `evaluation/evaluate.py::predict_by_split` (forward and backward MRR/Hits@k).
- All three optional Gumbel extensions together, including `--gumbel-topk-hop 2` with masked top-$k$ selection.
- `--random-anchor-selection` (A1), `--uniform-hop-weighting` (A5), `--fixed-lambda-p/-s 0` (A9/A10), and `--disable-link-graph` (A7, global-only candidates).
- `--num-hops 0` and `--num-hops 1` (hop-0-only and hop-0/1-only edge cases) and the `--anchor-budget 0` zero-candidate edge case.
- The full filtered-ranking checkpoint-selection protocol, including `--full-eval-every-n-epochs`, confirmed to run the expensive full-entity pass only on due epochs while `model_last.mdl` is still updated every epoch.
- Two-GPU DDP training via `torchrun --nproc_per_node=2`, confirmed to no longer raise the unused-parameter `RuntimeError`, with `MemoryGate`'s parameters confirmed to receive nonzero gradient norm every step via $\mathcal{L}_{\text{combined}}$, both with and without `--fixed-lambda-p`/`--fixed-lambda-s` active.

All configurations produced finite losses and metrics with no shape or NaN errors. In addition:

- `CandidatePoolBuilder` was checked directly against the synthetic training graph: `--num-hops 2` produced exactly the three local hop slots $\{0, 1, 2\}$ (hop-0: same head and relation with a different tail; hop-1: a genuinely different, one-edge-away head; hop-2: two edges away), and `ARPMModel.num_hop_slots`/`HopScorer`'s output width were confirmed to equal `num_hops + 1`.
- `_structural_memory`'s two safeguards were unit-tested against hand-crafted candidate grids: a query with an empty middle hop but populated neighboring hops (exact-zero weight on the empty hop confirmed even under an adversarially biased `G_hop`), a query with only global candidates, and a query with zero candidates at all — both of the latter confirmed to produce an exact-zero `m_struct` and `lambda_s`, including under a `--fixed-lambda-s 0.9` override.
- `--eta-combined 0` was run as the diagnostic control described above and confirmed to hold `lambda_p`/`lambda_s` at approximately their random-initialization values across epochs, in contrast to the default `--eta-combined 0.1` run, where both gates moved measurably from their initial values within the first epoch on the synthetic dataset. This is the intended minimal evidence that $\mathcal{L}_{\text{combined}}$ is necessary, not merely sufficient, for the gates to be "learned" in the sense the proposal assumes.

This confirms the implementation is mechanically correct. It does not substitute for a full training run on WN18RR or FB15k-237 to validate the method's empirical claims, which requires downloading `bert-base-uncased` and GPU time not available in this environment.

## Citation

If you use this implementation, please cite the underlying baseline it extends:

```bibtex
@article{yuan2025raakgc,
  title   = {Knowledge Graph Completion with Relation-Aware Anchor Enhancement},
  author  = {Yuan, Dong and Zhou, Sheng and Chen, Xin and Wang, Dan and Liang, Ke and Liu, Xinwang and Huang, Jun},
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume  = {39},
  number  = {14},
  pages   = {15239--15247},
  year    = {2025},
  doi     = {10.1609/aaai.v39i14.33672}
}
```

A citation for ARPM-KGC itself will be added here upon publication.

## Acknowledgements

The optional discrete (Gumbel-Softmax / Gumbel-Sigmoid) anchor, hop, and prototype gating extensions follow the straight-through, forward-hard/backward-soft design of Jang et al. (2017) and Maddison et al. (2017), applied in the same spirit as the per-node neighborhood-order selection of Lakzaei et al. (2025) [NOL-GAT].

## License

No license is currently specified for this repository. Add one (for example, MIT or Apache-2.0) before public release or redistribution.

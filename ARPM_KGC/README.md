# ARPM-KGC: Reference Implementation

This is a complete, standalone implementation of **Adaptive Relation-Aware
Prototype Memory for Knowledge Graph Completion (ARPM-KGC)**, as specified in
`ARPM_KGC_Proposal.tex`. It is built directly on top of the `Baseline/`
codebase (same dual-BERT encoder, same data format, same preprocessing, same
filtered-ranking evaluation protocol) so that the two can be trained and
compared like-for-like, but it is a fully separate package — nothing in
`ARPM_KGC/` imports from `Baseline/`.

```
ARPM_KGC/
  setting/      config.py, logger_config.py
  utils/        triplet.py, dict_hub.py, candidate_pool.py, doc.py, triplet_mask.py, utils.py
  model/        modules.py, models.py, trainer.py
  evaluation/   metric.py, predict.py, evaluate.py, eval_wiki5m_trans.py
  preprocess/   preprocess.py            (unchanged from Baseline — same data format)
  main.py
```

## 1. Proposal → code map

| Proposal section | What it defines | Where it lives |
|---|---|---|
| Sec. 3.2 Relation-Aware Candidate Memory | `A_local(h,r)`, `A_global(r)`, budget `M` | `utils/candidate_pool.py::CandidatePoolBuilder`, `utils/triplet.py::LinkGraph.get_hop_layers` |
| Sec. 3.3 Query-Conditioned Anchor Selection (RQ1) | `alpha_i`, weighted anchor set `W_q` | `model/models.py::ARPMModel._build_memory` |
| Sec. 3.3.1 Diversity Regularization | `L_div` | `model/modules.py::diversity_loss` |
| Sec. 3.4 Multi-Prototype Semantic Memory (RQ2) | `ProtoGen`, prototypes `p_k` | `model/modules.py::ProtoGen` |
| Sec. 3.5 Adaptive Structural Memory (RQ3) | `m^(l)`, `G_hop`, `beta_l`, `m_struct` | `model/models.py::ARPMModel._structural_memory`, `model/modules.py::HopScorer` |
| Sec. 3.6 Adaptive Memory-Aware Tail Prediction (RQ4) | `S_q, S_p, S_struct`, `G_lambda`, `S(t\|h,r)` | `model/models.py::ARPMModel.score_query/score_prototypes/score_struct/combined_score`, `model/modules.py::MemoryGate` |
| Sec. 3.7 Training Objective | `L = L_query + eta_p L_proto + eta_s L_struct + eta_div L_div` | `model/trainer.py::Trainer._compute_losses` |
| Sec. 3.8.1 Gumbel anchor gate (A11) | `c_i^ST` | `model/modules.py::gumbel_sigmoid_gate` |
| Sec. 3.8.2 Gumbel hop selection (A12) | `beta_l^ST`, top-k | `model/modules.py::gumbel_softmax_topk` |
| Sec. 3.8.3 Gumbel prototype gate (A13) | `omega_k^ST` | `model/modules.py::gumbel_sigmoid_slot_gate`, `PrototypeActivationScorer` |
| Sec. 5 Memory Gate Analysis | per-query `lambda_p, lambda_s` | dumped into `task_hrt_*.json` by `evaluation/evaluate.py::_save_prediction_details` |

`Example.vectorize(test=True)` (query, `E_0(h,r)`) and `Example.vectorize(test=False)`
(anchor, `E_0(h_i,r,t_i)`) already existed in `Baseline/utils/doc.py` for exactly
this purpose (they're what the baseline used for its own primary example and
its naive "related triplets"); ARPM-KGC reuses them unchanged and only
replaces *which* triplets get vectorized as anchors (`utils/candidate_pool.py`
instead of `Dataset.get_related_triplets`) and *how* they're combined
(`model/models.py` instead of a plain average).

## 2. Running it

Preprocessing is unchanged — reuse whatever `Baseline/data/<TASK>/*.txt.json` /
`entities.json` you already produced with `Baseline/preprocess/preprocess.py`
(or run `python -m ARPM_KGC.preprocess.preprocess --task wn18rr`, which
produces byte-identical output). From the directory that *contains*
`ARPM_KGC/`:

```bash
# Train (core model, all defaults from the proposal)
python -m ARPM_KGC.main --task wn18rr

# Evaluate the best checkpoint (filtered ranking: MRR, Hits@1/3/10/50)
python -m ARPM_KGC.evaluation.evaluate --task wn18rr --is-test

# Large-scale wiki5m_trans (sharded entity embeddings)
python -m ARPM_KGC.evaluation.eval_wiki5m_trans --task wiki5m_trans --is-test
```

Every `Baseline` flag (`--batch-size`, `--lr`, `--epochs`, `--pretrained-model`,
`--pooling`, `--use-amp`, `torchrun` DDP, ...) works identically. The
ARPM-KGC-specific hyperparameters (all with the defaults used above) are:

| Flag | Meaning | Default |
|---|---|---|
| `--num-hops` | `L`, structural hop distances | 2 |
| `--num-prototypes` | `K`, semantic prototypes, one of `{1,2,4,8}` | 4 |
| `--local-per-hop-budget` | max local anchors sampled per hop | 5 |
| `--global-budget` | max global anchors sampled from `T_r` | 10 |
| `--anchor-budget` | `M`, total candidate-pool cap | 20 |
| `--retrieval-temperature` | `tau_r` | 0.1 |
| `--proto-temperature` | `tau_p` | 0.1 |
| `--eta-proto`, `--eta-struct`, `--eta-div` | `eta_p, eta_s, eta_div` | 0.1, 0.1, 0.01 |
| `--disable-link-graph` | turn off local candidates entirely | off (link graph **on** by default — see §4) |
| `--checkpoint-metric` | metric used to pick the best checkpoint (§5) | `mrr` |
| `--full-eval-every-n-epochs` | cadence of the expensive full filtered-ranking validation (§5) | 1 |
| `--full-eval-batch-size` | batch size for that full validation pass | 256 |

## 3. Ablation study — every row is one flag

The proposal's ablation table (A1–A13, Full) is runnable directly, with no
code edits:

```bash
# A1  Random anchors instead of query-conditioned selection
python -m ARPM_KGC.main --task wn18rr --random-anchor-selection
# A2  No diversity regularization
python -m ARPM_KGC.main --task wn18rr --eta-div 0
# A3  Single prototype
python -m ARPM_KGC.main --task wn18rr --num-prototypes 1
# A4  Fixed vs. adaptive prototype count -- compare Full (fixed K, this default)
#     against A13 (--use-gumbel-proto), see note in §4
# A5  Fixed/uniform hop weighting
python -m ARPM_KGC.main --task wn18rr --uniform-hop-weighting
# A6  No global relation memory
python -m ARPM_KGC.main --task wn18rr --global-budget 0
# A7  No local structural memory (removes local anchors entirely, see §4)
python -m ARPM_KGC.main --task wn18rr --disable-link-graph
# A8  Fixed (non-adaptive) lambda_p, lambda_s
python -m ARPM_KGC.main --task wn18rr --fixed-lambda-p 0.5 --fixed-lambda-s 0.5
# A9  No S_p term
python -m ARPM_KGC.main --task wn18rr --fixed-lambda-p 0
# A10 No S_struct term
python -m ARPM_KGC.main --task wn18rr --fixed-lambda-s 0
# A11 Gumbel-Sigmoid anchor gating
python -m ARPM_KGC.main --task wn18rr --use-gumbel-anchor
# A12 Gumbel-Softmax hop selection
python -m ARPM_KGC.main --task wn18rr --use-gumbel-hop
# A13 Gumbel-Sigmoid prototype slot gating
python -m ARPM_KGC.main --task wn18rr --use-gumbel-proto
# Full
python -m ARPM_KGC.main --task wn18rr
```

## 4. Implementation notes and judgment calls

The proposal is precise about the model's math but necessarily silent on some
engineering choices; here is what was decided and why, so results are
interpretable and reproducible.

- **Temperature scaling is training-only.** `S_q`, `S_p`, `S_struct`, and the
  combined `S(t|h,r)` returned by `model/models.py` and used for *ranking*
  (evaluation, the Sec. 5 gate analysis) are always the raw, paper-exact
  similarities — no temperature, no margin. `model/trainer.py` applies
  Baseline's shared `exp(log_inv_t)` inverse-temperature (and, for `S_q`
  only, the additive margin + self-negative term) purely to sharpen the
  in-batch cross-entropy *losses*; since this is a positive monotonic
  rescaling applied identically inside each loss, it changes optimization
  dynamics but never the ranking any of the three scores would produce. This
  keeps training comparable to `Baseline`'s tuned SimKGC-style setup without
  contaminating the quantities the proposal actually defines.
- **False-negative masking (`triplet_mask`) is applied to `S_q`, `S_p`, and
  `S_struct` alike** during in-batch training — an unmasked false negative
  would corrupt all three losses, not just the query loss.
- **Pre-batch negatives are not implemented.** `Baseline` supports an
  optional pre-batch negative queue (`--pre-batch`, default `0`, i.e. off by
  default there too); it's orthogonal to ARPM-KGC's contribution and was
  dropped for clarity. Keep `--pre-batch 0` on the `Baseline` side when
  comparing.
- **`rerank_by_graph` is gone, on purpose.** `Baseline/utils/utils.py`'s
  post-hoc graph re-ranking heuristic is exactly what adaptive structural
  memory (Sec. 3.5), gated by the *learned* `lambda_s` (Sec. 3.6), is meant to
  replace with a principled, trainable mechanism — `evaluation/evaluate.py`
  now scores directly with `S(t|h,r)` instead.
- **Zero-candidate queries are handled explicitly, not silently patched.** If
  `CandidatePoolBuilder.build(...)` returns an empty list (isolated head,
  relation seen nowhere else), the model does **not** fall back to using the
  query's own true triple as an anchor — that would leak the label. Instead
  every candidate slot is the shared padding placeholder,
  `candidate_valid_mask` is all-`False` for that row, `alpha`/`L_div` are
  exactly `0`, and `m_struct`/prototypes are `0`-vectors (structural memory's
  `epsilon`, Sec. 3.5, exists precisely for this case). This was exercised
  directly (`--anchor-budget 0`) and produces finite, non-NaN losses.
- **`--disable-link-graph` removes local anchors from everything**, not just
  structural memory — i.e. it also removes local evidence from prototype
  construction (Sec. 3.4), since both draw from the same weighted anchor set
  `W_q`. The ablation table's A7 ("no local structural memory") and A10 ("no
  structural memory / drop `S_struct`") are, by the proposal's own Sec. 3.5
  definition, the same removal in principle (`m_struct` is defined to be
  built *only* from local anchors — there is no separate "global structural
  memory"). We've kept them distinct and both runnable: A10
  (`--fixed-lambda-s 0`) drops `S_struct` from the final score while local
  anchors still inform the prototypes; A7 (`--disable-link-graph`) removes
  local anchors from the pipeline altogether. Pick whichever isolates what
  you're trying to measure.
- **Adaptive prototype count `K_q`.** Sec. 3.4 explicitly specifies the core
  model uses a *fixed* `K in {1,2,4,8}`; Sec. 3.8.3 then introduces adaptive
  `K_q` only as the optional Gumbel extension A13. The ablation table's A4
  ("fixed prototype count instead of adaptive K") reads as if an
  always-adaptive `K` were part of "Full" — that isn't otherwise defined
  anywhere in the core model, so we've implemented Sec. 3.4 and Sec. 3.8.3
  literally: A4 is best read as the comparison between Full (fixed `K`, this
  default) and A13 (`--use-gumbel-proto`, adaptive `K_q`), not as a separate
  standalone flag.
- **Bilinear prototype scorer.** Sec. 3.4 leaves `S_P(a_i, q, k)` unspecified
  beyond naming it; it's implemented as a per-prototype bilinear form
  `u_ik = a_i^T W_k q / sqrt(d)` (`model/modules.py::ProtoGen`), the natural
  generalization of dot-product attention to `K` independent slots.
- **`G_hop(q,l)` and `G_lambda(q)`** are implemented as single linear layers
  whose `l`-th / `k`-th output channel *is* the corresponding score — the
  simplest faithful realization of "a function of `q` and a discrete index"
  that the proposal names without further specifying.
- **Candidate retrieval always reads the training graph only**
  (`utils/dict_hub.py::get_train_triplet_dict`, independent of
  `args.is_test`), so no validation/test label ever enters the candidate pool,
  at either training or evaluation time.
- **Efficiency.** Unlike `Baseline`'s Python `for`-loop over its (<=3)
  "related triplets", the whole candidate pipeline here — retrieval scoring,
  diversity, prototype attention, hop-specific aggregation — is vectorized
  over a padded `(batch, max_candidates)` grid with a single batched BERT
  call per pass; this was necessary anyway since ARPM-KGC's candidate budget
  `M` is much larger than Baseline's.

## 5. Checkpoint selection: real filtered-ranking metrics, not in-batch accuracy

Best-checkpoint selection uses the **same filtered-ranking protocol as final
evaluation** — forward + backward MRR/Hits@1/3/10/50 against the *full* entity
dictionary (`evaluation/evaluate.py::evaluate_predictor`, wrapped around the
in-memory model via `ARPMPredictor.from_model`) — not a cheap in-batch proxy.

This matters because in-batch accuracy (can the model pick the right tail out
of the ~batch_size other tails sitting in the same mini-batch?) and real
filtered ranking (can it beat the *entire* entity dictionary, with known
correct alternatives masked out?) are different tasks with different
difficulty; a model that looks great on the former can be middling on the
latter, so selecting on it can silently pick the wrong checkpoint. Concretely:

- `Trainer.eval_epoch` still runs every epoch and logs `Acc@1`/`Acc@3`/`loss`
  computed in-batch — cheap (one pass over `valid_loader`, no entity
  encoding), useful as a fast training-health signal, but explicitly logged as
  **"NOT used for checkpoint selection"** and never touches `best_metric`.
- `Trainer._compute_full_validation_metrics` runs the real protocol: it wraps
  the current in-memory model as a predictor, encodes every entity, and calls
  the exact same `evaluate_predictor` used by the standalone
  `evaluation/evaluate.py` script (just with `save_details=False`, so it
  doesn't spam per-query prediction files every epoch). Its output —
  `mean_rank`, `mrr`, `hit@1`, `hit@3`, `hit@10`, `hit@50` — is what decides
  `is_best`, via `--checkpoint-metric` (default `mrr`, the standard choice in
  this literature; `hit@1/3/10/50` are also selectable).
- Because encoding the whole entity dictionary every epoch is real work
  (fine for WN18RR/FB15k-237, prohibitive for wiki5m without sharding/caching
  a full pass), it only runs at epoch boundaries, and only every
  `--full-eval-every-n-epochs` epochs (default 1). Skipped epochs still save
  `model_last.mdl` (for resuming), they just never overwrite `model_best.mdl`.

No new files were added for this — `evaluation/predict.py::ARPMPredictor.from_model`
and `evaluation/evaluate.py::evaluate_predictor`/`save_details` are small,
reusable additions that both the trainer and the standalone eval scripts now
share, so there's exactly one code path computing MRR/Hits@k, not two that
could drift apart.

## 6. Validation performed

The full pipeline (candidate retrieval → anchor selection → diversity loss →
prototypes → structural memory → gating → combined score → training loss →
checkpointing → evaluation with filtered ranking) was exercised end-to-end on
a small synthetic dataset with a randomly-initialized, fully local (no
internet) tiny BERT, covering: a full `train_loop` with checkpoint save/load,
`evaluation/evaluate.py::predict_by_split` (forward + backward MRR/Hits@k),
all three optional Gumbel extensions together, `--disable-link-graph`
(global-only candidates), the `--anchor-budget 0` zero-candidate edge case,
**and the full-filtered-ranking checkpoint selection of §5** (including
`--full-eval-every-n-epochs`, verified to run the expensive full-entity pass
only on due epochs while `model_last.mdl` is still updated every epoch) — all
produced finite losses/metrics with no shape or NaN errors. This confirms the
implementation is mechanically correct; it does **not** substitute for a real
training run on WN18RR/FB15k-237 to validate the proposal's empirical claims,
which requires downloading `bert-base-uncased` and GPU time neither of which
were available in this environment.

# ARPM-KGC: Reference Implementation

This is a complete, standalone implementation of **Adaptive Relation-Aware
Prototype Memory for Knowledge Graph Completion (ARPM-KGC)**.

```
ARPM_KGC/
  setting/      config.py, logger_config.py
  utils/        triplet.py, dict_hub.py, candidate_pool.py, doc.py, triplet_mask.py, utils.py
  model/        modules.py, models.py, trainer.py
  evaluation/   metric.py, predict.py, evaluate.py, eval_wiki5m_trans.py
  preprocess/   preprocess.py            
  main.py
```

## 1. Proposal → code map

| Proposal section | What it defines | Where it lives |
|---|---|---|
| Sec. 3.2 Relation-Aware Candidate Memory | `A_local(h,r)`, `A_global(r)`, budget `M` | `utils/candidate_pool.py::CandidatePoolBuilder`, `utils/triplet.py::LinkGraph.get_hop_layers` |
| Sec. 3.3 Query-Conditioned Anchor Selection (RQ1) | `alpha_i`, weighted anchor set `W_q` | `model/models.py::ARPMModel._build_memory` |
| Sec. 3.3.1 Diversity Regularization | `L_div` | `model/modules.py::diversity_loss` |
| Sec. 3.4 Multi-Prototype Semantic Memory (RQ2) | `ProtoGen`, prototypes `p_k` | `model/modules.py::ProtoGen` |
| Sec. 3.5 Adaptive Structural Memory (RQ3) | `m^(l)`, `G_hop`, `beta_l`, `m_struct` (plus empty-hop / no-local-anchor safeguards, §4) | `model/models.py::ARPMModel._structural_memory`, `model/modules.py::HopScorer` |
| Sec. 3.6 Adaptive Memory-Aware Tail Prediction (RQ4) | `S_q, S_p, S_struct`, `G_lambda`, `S(t\|h,r)` | `model/models.py::ARPMModel.score_query/score_prototypes/score_struct/combined_score`, `model/modules.py::MemoryGate` |
| Sec. 3.7 Training Objective | `L = L_query + eta_p L_proto + eta_s L_struct + eta_div L_div` | `model/trainer.py::Trainer._compute_losses` |
| Sec. 3.8.1 Gumbel anchor gate (A11) | `c_i^ST` | `model/modules.py::gumbel_sigmoid_gate` |
| Sec. 3.8.2 Gumbel hop selection (A12) | `beta_l^ST`, top-k | `model/modules.py::gumbel_softmax_topk` |
| Sec. 3.8.3 Gumbel prototype gate (A13) | `omega_k^ST` | `model/modules.py::gumbel_sigmoid_slot_gate`, `PrototypeActivationScorer` |
| Sec. 5 Memory Gate Analysis | per-query `lambda_p, lambda_s` | dumped into `task_hrt_*.json` by `evaluation/evaluate.py::_save_prediction_details` |

`Example.vectorize(test=True)` (query, `E_0(h,r)`) and `Example.vectorize(test=False)`
(anchor, `E_0(h_i,r,t_i)`); ARPM-KGC reuses them unchanged and only
replaces *which* triplets get vectorized as anchors.

## 2. Running it

Preprocessing
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

Every flag (`--batch-size`, `--lr`, `--epochs`, `--pretrained-model`,
`--pooling`, `--use-amp`, `torchrun` DDP, ...) works identically. The
ARPM-KGC-specific hyperparameters (all with the defaults used above) are:

| Flag | Meaning | Default |
|---|---|---|
| `--num-hops` | `N`, max graph distance beyond hop-0 (yields N+1 local categories: hop-0..hop-N) | 2 |
| `--num-prototypes` | `K`, semantic prototypes, one of `{1,2,4,8}` | 4 |
| `--local-per-hop-budget` | max local anchors sampled per hop (incl. hop-0) | 5 |
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
  shared `exp(log_inv_t)` inverse-temperature (and, for `S_q`
  only, the additive margin + self-negative term) purely to sharpen the
  in-batch cross-entropy *losses*; since this is a positive monotonic
  rescaling applied identically inside each loss, it changes optimization
  dynamics but never the ranking any of the three scores would produce.
- **False-negative masking (`triplet_mask`) is applied to `S_q`, `S_p`, and
  `S_struct` alike** during in-batch training — an unmasked false negative
  would corrupt all three losses, not just the query loss.
- **Pre-batch negatives are not implemented.**
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
- **Efficiency.** the whole candidate pipeline here — retrieval scoring,
  diversity, prototype attention, hop-specific aggregation — is vectorized
  over a padded `(batch, max_candidates)` grid with a single batched BERT
  call per pass; this was necessary anyway since ARPM-KGC's candidate budget
  `M` is much larger than Baseline's.
- **Local hop 0 is "same head & relation", not "nearest graph neighbor" —
  and `--num-hops N` yields N+1 categories, not N.** Local candidates span
  hop-0 (other valid tails for the *exact same* `(h, r)` pair — graph
  distance 0, no traversal — tightest possible evidence, most useful for
  one-to-many relations), then hop-1 (graph distance 1, i.e. direct/1-edge
  neighbors), ..., hop-N (graph distance N): `--num-hops 2` therefore builds
  *three* structural-memory slots, hop-0/1/2, not two. `HopScorer` and
  `m^(l)` are allocated `num_hops + 1` slots accordingly
  (`ARPMModel.num_hop_slots`, `model/models.py`). Global candidates, and
  padding slots in `utils/doc.py::collate`, are tagged `NO_HOP = -1`
  (`utils/candidate_pool.py`) precisely so they can never be mistaken for a
  genuine local hop-0 candidate — reusing `0` as that sentinel would have
  silently collided with this indexing. `_global_candidates` also excludes
  any pair sharing the query's head (not just its exact true tail), since
  that's already covered by the dedicated hop-0 category — keeping "global"
  strictly meaning "a different head" so the two categories don't overlap.
- **Empty-hop and no-local-anchor safeguards in `_structural_memory`
  (Sec. 3.5), both parameter-free:**
  1. *Some hops empty, not all.* The hop-selection weighting — the dense
     `softmax(G_hop(q))`, its `--use-gumbel-hop` (A12) counterpart, and even
     the `--uniform-hop-weighting` (A5) ablation — is masked so a hop with
     zero local anchors for a given query gets *exactly* zero weight,
     instead of relying on `G_hop` to learn that on its own. Verified
     adversarially: forcing `G_hop` to output a very large logit for a
     deliberately-empty hop still yields exactly `0` weight for it after
     masking.
  2. *No local anchor at any hop (including hop-0).* `m_struct` is hard-set
     to the zero vector for that query (via `torch.where`, an element-wise
     *replace* rather than a multiplicative mask — the latter would let a
     NaN from a degenerate all-hops-masked softmax leak through as
     `0 * NaN = NaN`), and `forward()` hard-overrides `lambda_s <- 0` for
     that query, *after* any `--fixed-lambda-s` ablation override (so "no
     evidence" always wins, even under a deliberately fixed non-adaptive
     gate — see §3, A8/A10). Verified directly, including that the override
     survives `--fixed-lambda-s 0.9`.

  Fixing safeguard 2 surfaced a related pre-existing off-by-one in
  `CandidatePoolBuilder._local_candidates`/`_same_head_candidates`: the
  per-hop budget was checked *after* appending a candidate, so
  `--local-per-hop-budget 0` still let one candidate per non-empty hop
  through. The check now runs before appending, so a `0` budget means
  exactly `0` local candidates, as documented. `--num-hops 0` is also a
  valid, meaningful configuration under the corrected indexing: hop-0
  (same head & relation) only, with no graph traversal at all — a more
  surgical restriction than `--disable-link-graph` (A7), which removes
  *every* local category, hop-0 included.
- **Known limitation, not (yet) fixed: `A(h,r) = A_local(h,r) U A_global(r)`
  is currently a concatenation, not a true set union.** `_global_candidates`
  excludes any pair sharing the query's head (deduplicating against hop-0),
  but a triple reached via graph traversal for hop `l >= 1` can still also be
  independently sampled into the global pool, so it may appear twice with
  two different tags (once local/hop-`l`, once global). No correctness
  impact -- the local copy still feeds structural memory correctly, the
  extra global copy just adds slightly more retrieval/prototype weight to
  that one anchor -- but worth knowing if you inspect per-query candidate
  pools directly. Flagging rather than silently patching this since it
  wasn't part of what was asked; happy to dedupe properly (e.g. tracking
  seen `(head_id, tail_id)` pairs per query) if wanted.

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
all three optional Gumbel extensions together (including `--gumbel-topk-hop 2`
with a masked top-k selection), `--random-anchor-selection` (A1),
`--uniform-hop-weighting` (A5), `--fixed-lambda-p/-s 0` (A9/A10),
`--disable-link-graph` (A7, global-only candidates), `--num-hops 0` and
`--num-hops 1` (hop-0-only and hop-0/1-only edge cases), the `--anchor-budget
0` zero-candidate edge case, and the full-filtered-ranking checkpoint
selection of §5 (including `--full-eval-every-n-epochs`, verified to run the
expensive full-entity pass only on due epochs while `model_last.mdl` is still
updated every epoch) — all produced finite losses/metrics with no shape or
NaN errors. In addition:
- `CandidatePoolBuilder` was checked directly against the synthetic training
  graph: `--num-hops 2` was confirmed to produce exactly the three local hop
  slots `{0, 1, 2}` (hop-0 = same head & relation with a different tail,
  hop-1 = a genuinely different, 1-edge-away head, hop-2 = 2 edges away), and
  `ARPMModel.num_hop_slots` / `HopScorer`'s output width were confirmed to
  equal `num_hops + 1`.
- `_structural_memory`'s two safeguards (§4) were unit-tested directly
  against hand-crafted candidate grids (re-run under the corrected `L =
  num_hops + 1` shape): a query with an empty middle hop but populated
  neighboring hops (verified exact-zero weight on the empty hop even when
  `G_hop` is adversarially biased toward it), a query with only global
  candidates, and a query with zero candidates at all (both verified to
  produce an exact-zero `m_struct` and `lambda_s`, including under a
  `--fixed-lambda-s 0.9` override).

This confirms the implementation is mechanically correct; it does **not**
substitute for a real training run on WN18RR/FB15k-237 to validate the
proposal's empirical claims, which requires downloading `bert-base-uncased`
and GPU time neither of which were available in this environment.

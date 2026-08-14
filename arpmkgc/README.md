# ARPM-KGC

Independent, from-scratch implementation of **ARPM-KGC** (Adaptive
Relation-Aware Prototype Memory for Knowledge Graph Completion), reimplementing
the architecture and experimental plan described in the accompanying research
proposal. This package is self-contained: it does not import anything from
the separate baseline codebase (`ours/`), so the two can be used side by side
without interference.

Every optional mechanism in the proposal (Sections 4.4-4.13) is a config
flag, so every row of the proposal's ablation study (Table 3) is a preset
away — no code edits required.

```
python -m arpmkgc.main train --ablation full --task wn18rr
python -m arpmkgc.main train --ablation A9 --task fb15k237 --epochs 20
python -m arpmkgc.main evaluate --ablation full --task wn18rr \
    --eval-model-path data/WN18RR/checkpoint/full/model_best.mdl --eval-split test
python -m arpmkgc.main list-ablations
```

## 1. Package layout

```
arpmkgc/
  config.py            ARPMConfig -- every proposal knob, one dataclass
  ablations.py          Table 3 presets (baseline, A1-A13, full)
  candidates.py          Sec 4.3: local (hop-based) + global candidate pool
  model.py                ARPMKGCModel: wires every module behind config flags
  losses.py                Sec 4.11: L_final, L_p, L_retr = L_rel + eta_div*L_div
  masking.py                In-batch false-negative / self-negative masks
  metrics.py                  accuracy() + filtered ranking (MRR, Hits@k)
  trainer.py                   Training loop, checkpointing, optional DDP
  evaluate.py                   Filtered-ranking eval (forward+backward), Sec 8 logging
  predict.py                     Checkpoint loading / inference wrapper
  preprocess.py                   WN18RR / FB15k-237 raw-data -> JSON
  main.py                          CLI entry point
  utils.py, logging_utils.py        Generic infra (independent of `ours`)

  data/
    entities.py    EntityDict            triplets.py  TripletDict, RelationVocab
    graph.py       LinkGraph (hop BFS)   dict_hub.py  process-wide singletons
    dataset.py     Example/Dataset/Collator (tokenization for E0, E1)

  modules/
    encoders.py    E0 (query/anchor) + E1 (entity) BERT towers  -- Sec 4.2
    retrieval.py   S_A anchor scorer, alpha, L_div, 4.13.1 gate -- Sec 4.4
    prototype.py   ProtoGen, query-prototype interaction, 4.13.3 -- Sec 4.5, 4.7
    structural.py  hop aggregation, G_hop, 4.13.2               -- Sec 4.6
    fusion.py      G_mem, G_q, G_lambda                          -- Sec 4.8-4.10
    gumbel.py      shared Gumbel-Softmax/-Sigmoid ST machinery   -- Sec 4.13
```

## 2. Architecture -> code map

| Proposal | Symbol | Code |
|---|---|---|
| 4.2 Encoders | `q = E0(h,r)`, `a_i = E0(h_i,r,t_i)`, `e_t = E1(t)` | `modules/encoders.py: DualTowerEncoder` |
| 4.3 Candidate memory | `A_local^(l)(h,r)`, `A_global(r)`, budget `M` | `candidates.py: CandidatePoolBuilder` |
| 4.4 Anchor selection | `S_A`, `alpha_i = softmax(s_i/tau_r)` | `modules/retrieval.py: AnchorSelector` |
| 4.4.1 Diversity | `L_div`, `L_retr = L_rel + eta_div L_div` | `AnchorSelector.diversity_loss`, `losses.py` |
| 4.5 Prototype memory | `ProtoGen`, `rho_ik`, `p_k` | `modules/prototype.py: PrototypeGenerator` |
| 4.6 Structural memory | `m^(l)`, `G_hop`, `beta_l`, `m_struct` | `modules/structural.py: StructuralMemory` |
| 4.7 Query-prototype interaction | `q_p`, `gamma_k`, `m_p` | `modules/prototype.py: QueryPrototypeInteraction` |
| 4.8 Memory fusion | `G_mem(m_p, m_struct) = m` | `modules/fusion.py: MemoryFusion` |
| 4.9 Query-memory fusion | `G_q(q_p, m) = q_e` | `modules/fusion.py: QueryMemoryFusion` |
| 4.10 Prediction | `S_q`, `S_p` (log-sum-exp), `lambda_mem`, `S` | `model.py: ARPMKGCModel.score` |
| 4.11 Objective | `L_final`, `L_p`, `L = L_final + eta_p L_p + eta_r L_retr` | `losses.py` |
| 4.12 Inference | mirrors training, no label leakage | `evaluate.py`, `predict.py` |
| 4.13.1 Discrete anchors | Gumbel-Sigmoid keep/drop gate | `retrieval.py` + `modules/gumbel.py` |
| 4.13.2 Discrete hops | Gumbel-top-k hop selection | `structural.py` + `modules/gumbel.py` |
| 4.13.3 Adaptive K | Gumbel-Sigmoid slot gating | `prototype.py: SlotGate` + `modules/gumbel.py` |

## 3. Config flags <-> Table 3 ablations

`ablations.py` turns each row of Table 3 into an `ARPMConfig` preset. All
presets start from `full_config()` (every core mechanism on, every 4.13
extension off) and flip exactly the flag(s) below:

| Row | Flag(s) changed |
|---|---|
| Baseline | see `baseline_faithful_config()` -- single unconditional anchor-average memory, no scoring/gating/structural adaptivity (best-effort reconstruction; the cited paper's own code isn't available to us -- see note below) |
| A1 | `anchor_selection_mode = "random"` |
| A2 | `use_diversity_regularization = False` |
| A3 | `num_prototypes = 1` |
| A4 | `prototype_gating_mode = "fixed"` (control for A13; equals `full_config()`) |
| A5 | `hop_selection_mode = "uniform"` |
| A6 | `use_global_candidates = False` |
| A7 | `use_local_candidates = False` |
| A8 | `use_adaptive_memory_gate = False` |
| A9 | `use_prototype_memory = False` |
| A10 | `use_structural_memory = False` |
| A11 | `anchor_gating_mode = "gumbel_sigmoid"` |
| A12 | `hop_selection_mode = "gumbel_softmax"` |
| A13 | `prototype_gating_mode = "gumbel_sigmoid_slots"` |

Section 7.3's four-way baseline comparison maps onto this codebase as: (1)
reported external numbers and (2) the released baseline implementation are
both out of scope here (item 2 is literally the separate `ours/` package);
(3) the "faithful reconstruction" and (4) "our pipeline with the original
anchor mechanism" are, by construction, the same object in a unified
codebase -- both are served by the single `baseline` preset above.

## 4. Documented ambiguity resolutions

The proposal is a research plan, not a fully pinned-down spec. Three places
required a judgment call; each is called out in the relevant module
docstring as well as here:

1. **Hop scorer input (`q` vs `q_p`)** -- Sec 4.6's prose says `G_hop` scores
   hops using `q_p`, "the query representation after prototype interaction."
   But Algorithm 1 computes structural memory (line 15, using raw `q`)
   *before* `q_p` exists (line 19). We default to the executable Algorithm 1
   reading (`hop_score_uses_prototype_query=False`) and expose the prose
   reading as an opt-in flag.
2. **A6/A7 vs A9/A10** -- A6 ("no global relation memory") and A7 ("no local
   structural memory") are read as *candidate-source* ablations (Sec 4.3:
   remove `A_global(r)` / `A_local(h,r)` from the pool). A9/A10 are read as
   the two *fusion-branch* ablations at `G_mem` (Sec 4.8) -- the pair
   Sec 7.4 explicitly says "test whether the two memory sources are
   complementary." A7's removal of local candidates is also the most literal
   way to realize "no local structural memory," since structural memory is
   local-only by construction.
3. **A4 vs A13** -- since the core model already uses a fixed K (Sec 4.5.1),
   A4 ("fixed prototype count instead of adaptive K") is definitionally the
   same config as `full_config()`; it is kept as a separately-named preset
   purely so it reads clearly as A13's control in a results table.

## 5. Design decisions borrowed from standard KGC practice (not specified by the proposal)

The proposal focuses on the memory architecture, not every engineering
detail. Where it is silent, this codebase makes an explicit, documented
choice rather than silently guessing:

- **In-batch contrastive training** with additive margin, false-negative
  batch masking, and an optional self-negative (`config.use_self_negative`)
  -- standard practice for dual-encoder KGC, not specific to any baseline.
- **`sim(.,.)`** is cosine similarity scaled by a single learnable global
  inverse temperature (`log_inv_t`), applied consistently inside both `S_q`
  and the per-prototype similarities feeding `S_p`'s log-sum-exp, so the two
  terms are on a comparable scale before being combined via `lambda_mem`.
- **Relation embeddings** (`r` in `S_A`, `G_hop`, `G_K`, `G_lambda`) are four
  independent `nn.Embedding` tables, one per scorer, rather than a single
  shared table -- each scorer can specialize its own notion of relation
  semantics. This is a reasonable default, not a proposal requirement.
- **E0/E1 are two separately-initialized BERT towers** (`tie_encoders=False`
  by default); `q = E0(h, r)` never sees the true tail, at train or test
  time, matching Sec 4.2's definition and Sec 4.12's "no target-tail
  leakage" requirement literally.
- **Candidate retrieval always draws from the training-split link graph and
  triplet index**, regardless of whether the query being scored is a train,
  valid, or test example (Sec 4.12's inference procedure mirrors training
  exactly except for label use) -- see `data/dict_hub.py`'s
  `get_link_graph`/`get_train_triplet_dict` and their docstrings.
- **L_rel** (explicit retrieval supervision) defaults to **off**, per the
  proposal's own note that retrieval can be learned indirectly through
  `L_final`/`L_p`. When enabled (`use_retrieval_supervision=True`), it uses
  a simple weak-label heuristic (an anchor is a positive iff its tail equals
  the query's gold tail) documented in `retrieval.py`.
- **`eta_r` (`retrieval_loss_weight`) scales the whole `L_retr = L_rel +
  eta_div * L_div` sum**, not just `L_rel` -- so diversity regularization
  (on by default) has a non-zero effect on the total loss even when
  `L_rel` itself is disabled (which it is, by default). Default `eta_r=1.0`.

## 6. Data preparation

```
python -m arpmkgc.preprocess --task wn18rr    --raw-dir <dir with train/valid/test.txt + wordnet-mlj12-definitions.txt>
python -m arpmkgc.preprocess --task fb15k237  --raw-dir <dir with train/valid/test.txt + FB15k_mid2{name,description}.txt>
```

Writes `<split>.txt.json`, `entities.json`, and `relations.json` (the
relation vocabulary used by every `r`-conditioned scorer) into
`data/<TASK>/` by default. `relations.json` includes the `"inverse <r>"`
relation for every forward relation, matching the convention used when
examples are loaded (`data/triplets.py: reverse_triplet`), so backward
(tail->head) queries share the same relation-embedding table.

## 7. A note on validation

This environment has no GPU/torch/network access, so the implementation was
validated with a hand-written numpy-backed stub of the torch/transformers
API (not shipped as part of the package) that let every code path actually
execute: all 15 ablation presets ran forward + loss end-to-end without NaNs
on synthetic data, a full `Trainer` epoch completed with checkpointing, a
saved checkpoint round-tripped through `ARPMPredictor`, the filtered-ranking
evaluator produced sane forward/backward/averaged metrics, both CLI commands
(`train`, `evaluate`) worked, and both dataset preprocessors ran correctly
on tiny synthetic raw fixtures. That exercise caught and fixed several real
bugs (an all-padding attention-mask NaN hazard, a DDP unused-parameter gap,
a checkpoint-vs-runtime config mismatch in `evaluate.py`, and a CLI flag
being silently dropped) -- but it does not substitute for an actual run
against real WN18RR/FB15k-237 data on a GPU, which is the natural next step.

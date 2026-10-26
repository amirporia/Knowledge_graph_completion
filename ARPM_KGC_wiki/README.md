# ARPM-KGC: Adaptive Relation-Aware Prototype Memory for Knowledge Graph Completion

Reference implementation of **ARPM-KGC**, a text-based knowledge graph completion (KGC) model that extends the relation-aware anchor framework of Yuan et al. (2025) [RAA-KGC]. For a query $(h, r, ?)$ it retrieves a bounded pool of relation-aware anchors from the **training graph only**, and turns that pool into two compact memories: a **multi-prototype semantic memory** and a **hop-aware structural memory**. Both are scored against every candidate entity and added, with constant weights tuned on the validation split, to the dual-encoder score.

Evaluation follows the standard filtered-ranking protocol (MRR, Hits@1/3/10/50, mean rank) on WN18RR, FB15k-237 and Wikidata5M.

## Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Repository structure](#repository-structure)
- [Data preparation](#data-preparation)
- [Usage](#usage)
- [Hyperparameters and flags](#hyperparameters-and-flags)
- [Score coefficients](#score-coefficients)
- [Ablation study](#ablation-study)
- [Checkpoints and model selection](#checkpoints-and-model-selection)
- [Outputs](#outputs)
- [Implementation notes](#implementation-notes)
- [Method-to-code mapping](#method-to-code-mapping)
- [Known limitations](#known-limitations)
- [Citation](#citation)

## Overview

Given a query $(h, r, ?)$, ARPM-KGC builds a candidate pool $\mathcal{A}(h,r) = \mathcal{A}_{\text{local}}(h,r) \cup \mathcal{A}_{\text{global}}(r)$:

- **Local anchors** come from the neighbourhood of $h$ in the training graph, tagged with a hop slot: hop-0 = other known tails of the same $(h,r)$, hop-$\ell$ = a head at graph distance $\ell$ that has relation $r$ (up to `--num-hops`).
- **Global anchors** are random relation-$r$ training pairs with a different head (tagged `NO_HOP = -1`).

Each anchor is represented by its tail embedding $E_1(t_i)$. Over this pool the model makes three query-conditioned decisions:

1. **Anchor selection (RQ1).** Attention weights $\alpha_i = \mathrm{softmax}_i\big(a_i^\top W q / \tau_r\big)$ decide which anchors matter for this query.
2. **Prototype organisation (RQ2).** `ProtoGen` builds $K$ query-conditioned prototypes $p_k$ from the weighted anchors (bilinear attention per slot) instead of one averaged vector.
3. **Structural aggregation (RQ3).** Per-hop memories $m^{(\ell)}$ are combined with learned, hop-availability-masked weights $\beta_\ell = \mathrm{softmax}(G_{\text{hop}}(q))$ into $m_{\text{struct}}$.

The final ranking score is

$$
S(t \mid h,r) \;=\; S_q(t) \;+\; c_h\, S_{\text{hrta}}(t) \;+\; c_p\, S_p(t) \;+\; c_s\, S_{\text{struct}}(t)
$$

| Term | Definition |
| --- | --- |
| $S_q$ | $\cos(q, e_t)$, the (head, relation) query against the entity embedding |
| $S_{\text{hrta}}$ | $\cos(q_{\text{hrta}}, e_t)$, the RAA-KGC anchor-enhanced query (mean of $E_0(h,r,t_i)$ over up to `--anchor-num` hop-0 anchors; falls back to $q$) |
| $S_p$ | $\tau_p\big(\log\sum_k \exp(\cos(p_k, e_t)/\tau_p) - \log K\big)$, a soft maximum over prototypes |
| $S_{\text{struct}}$ | $\cos(m_{\text{struct}}, e_t)$ |

**$c_h, c_p, c_s$ are constants, not learned.** They are chosen by a grid search on the *validation* split (`--tune-coefs`) or given explicitly (`--coef-h/--coef-p/--coef-s`). With $c_h = 1,\ c_p = c_s = 0$ the model reduces exactly to the RAA-KGC inference rule, which makes the baseline a special case of the same checkpoint. Queries whose memory is empty contribute exactly 0 for the missing term, so they degrade gracefully to the query/RAA score.

> **Version note.** An earlier revision of this repository described a learned memory gate $G_\lambda$ (with $\lambda_p, \lambda_s$), an auxiliary $\mathcal{L}_{\text{combined}}$, and Gumbel-based discrete gating. None of these exist in the current code. Checkpoints containing `memory_gate.*` weights come from that revision and cannot be loaded.

## Requirements

- Python 3.8+
- PyTorch 2.x (2.3+ recommended; the code uses `torch.amp.autocast` and `device_id` in `init_process_group`)
- `transformers`, `tqdm`

No versions are pinned. A CUDA GPU is strongly recommended; multi-GPU training works either through `torchrun` (DDP) or, when launched with plain `python` on a multi-GPU machine, `nn.DataParallel`.

```bash
pip install torch transformers tqdm
```

## Repository structure

The Python package folder is `ARPM_KGC_DataParallel_original/`; all commands below are run **from the directory that contains it** (the repository root), with `data/` next to it. (You may rename the folder, just replace the name in the commands.)

```
Knowledge_graph_completion/
  data/<task>/                      # created by you / by preprocessing
  ARPM_KGC_DataParallel_original/
    setting/      config.py, logger_config.py
    utils/        triplet.py, dict_hub.py, candidate_pool.py, doc.py, triplet_mask.py, utils.py
    model/        modules.py, models.py, trainer.py
    evaluation/   metric.py, predict.py, evaluate.py, eval_wiki5m_trans.py
    preprocess/   preprocess.py
    main.py
```

## Data preparation

Put the raw files under `data/<task>/` at the repository root:

| Task | Required raw files |
| --- | --- |
| `wn18rr` | `train.txt`, `valid.txt`, `test.txt`, `wordnet-mlj12-definitions.txt` |
| `fb15k237` | `train.txt`, `valid.txt`, `test.txt`, `FB15k_mid2description.txt`, `FB15k_mid2name.txt` |
| `wiki5m_trans` / `wiki5m_ind` | `train/valid/test.txt`, `wikidata5m_entity.txt`, `wikidata5m_relation.txt`, `wikidata5m_text.txt` (see `download_wikidata5m.sh` in `SimKGC_Restructured/scripts/` or `HaSa_Restructured/scripts/`) |

Then run:

```bash
python -m ARPM_KGC_DataParallel_original.preprocess.preprocess --task wn18rr
# also: fb15k237, wiki5m_trans, wiki5m_ind
```

This writes `train/valid/test.txt.json`, `entities.json` and `relations.json` into `data/<task>/`, which `utils/dict_hub.py` and `utils/doc.py` read. The preprocessing output is shared with the other packages in this repository.

## Usage

### Train (single GPU)

The command below uses the settings of our WN18RR experiments:

```bash
python -m ARPM_KGC_DataParallel_original.main \
  --task wn18rr \
  --model-dir data/wn18rr/checkpoint_arpm \
  --pretrained-model bert-base-uncased --pooling mean --max-num-tokens 50 \
  --epochs 20 --batch-size 32 --lr 5e-5 --memory-lr 1e-4 --wd 1e-4 \
  --warmup 400 --grad-clip 10 --t 0.05 --additive-margin 0.02 --alpha 0.2 \
  --anchor-num 4 --num-hops 2 --num-prototypes 4 \
  --local-per-hop-budget 5 --global-budget 10 --anchor-budget 25 \
  --eta-proto 0.1 --eta-struct 0.1 --eta-pdiv 0.01 --eta-div 0 \
  --rerank-n-hop 2 --neighbor-weight 0.05 \
  --use-amp --workers 4 --seed 42
```

Settings used for FB15k-237 in our evaluation runs differ in the memory budget: `--num-prototypes 2 --global-budget 5 --anchor-budget 20 --num-hops 2 --rerank-n-hop 2` (training lr/epochs for FB15k-237 should mirror your baseline runs).

`--finetune-t` and `--use-self-negative` are on by default (disable with `--no-finetune-t`, `--no-use-self-negative`).

### Train (multi-GPU)

```bash
torchrun --nproc_per_node=2 -m ARPM_KGC_DataParallel_original.main --task wn18rr <same flags>
```

`--batch-size` is **per process** under `torchrun` (each rank draws in-batch negatives from its own batch). With plain `python` on a multi-GPU machine, `nn.DataParallel` is used and `--batch-size` is the global batch.

### Resume

Add `--resume` (optionally `--resume-path`; default `<model-dir>/model_last.mdl`). Resumption is at **epoch granularity**: the model, optimizer, scheduler and AMP scaler are restored and training continues with the next epoch; a partially finished epoch is not replayed. This makes it easy to train on time-limited machines: stop the job at any point, keep `model_last.mdl`, and restart with `--resume`.

### Evaluate

Evaluation reads the checkpoint's own training arguments (model, candidate-pool settings, link graph), but `--max-num-tokens`, `--neighbor-weight`, `--rerank-n-hop`, `--batch-size` come from the command line, so pass the **same values as in training**.

```bash
EVAL_FLAGS="--task wn18rr --eval-model-path data/wn18rr/checkpoint_arpm/model_last.mdl \
  --pretrained-model bert-base-uncased --pooling mean --max-num-tokens 50 \
  --batch-size 32 --is-test --seed 42 --workers 4 \
  --rerank-n-hop 2 --neighbor-weight 0.05"

# 1) tune (c_h, c_p, c_s) on the VALIDATION split (refuses a path containing "test").
#    Writes best_coefs.json next to the checkpoint.
python -m ARPM_KGC_DataParallel_original.evaluation.evaluate $EVAL_FLAGS --tune-coefs \
  --coef-h-grid 0 0.5 1 1.5 --coef-p-grid 0 0.1 0.25 0.5 1 --coef-s-grid 0 0.1 0.25 0.5 1

# 2) evaluate the TEST split once. best_coefs.json is picked up automatically
python -m ARPM_KGC_DataParallel_original.evaluation.evaluate $EVAL_FLAGS \
  --valid-path data/wn18rr/test.txt.json
```

Coefficient resolution order (`evaluation/evaluate.py::resolve_coefs`): (1) `--tune-coefs`; (2) explicit `--coef-h/--coef-p/--coef-s` (any unset one falls back to `DEFAULT_COEFS = (1.0, 0.1, 0.1)`); (3) `best_coefs.json` next to the checkpoint when no coefficient was given; (4) `DEFAULT_COEFS`. To bypass the file and fix the weights, pass all three explicitly:

```bash
python -m ARPM_KGC_DataParallel_original.evaluation.evaluate $EVAL_FLAGS \
  --valid-path data/wn18rr/test.txt.json --coef-h 1 --coef-p 0.3 --coef-s 0.2
```

### Large-scale `wiki5m_trans`

```bash
python -m ARPM_KGC_DataParallel_original.evaluation.eval_wiki5m_trans --task wiki5m_trans --is-test \
  --eval-model-path <ckpt> --valid-path data/wiki5m_trans/test.txt.json
```

Entity embeddings are computed and stored in shards of 1M entities (`shard_<i>` in `--model-dir`) and then concatenated on the GPU.

## Hyperparameters and flags

Defaults are those of `setting/config.py`.

### Candidate pool and memory

| Flag | Meaning | Default |
| --- | --- | --- |
| `--anchor-num` | Number of hop-0 anchors for the RAA anchor-enhanced query $q_{\text{hrta}}$ (0 disables $S_{\text{hrta}}$) | 4 |
| `--num-hops` | $N$: max graph distance of local anchors; yields $N+1$ hop slots (hop-0 … hop-$N$) | 5 |
| `--num-prototypes` | $K$ semantic prototypes (recommended $\in\{1,2,4,8\}$) | 4 |
| `--local-per-hop-budget` | Max local anchors per hop slot (incl. hop-0) | 5 |
| `--global-budget` | Max global (different-head) relation-$r$ anchors | 5 |
| `--anchor-budget` | $M$: cap on the merged pool (hop-0 anchors are never dropped by it) | 20 |
| `--retrieval-temperature` | $\tau_r$ of the anchor attention $\alpha_i$ | 0.1 |
| `--proto-temperature` | $\tau_p$ of the log-sum-exp pooling in $S_p$ | 0.1 |
| `--proto-attn-temperature` | Temperature of the per-prototype attention inside `ProtoGen` | 0.5 |
| `--eps-struct` | $\epsilon$ in the per-hop weighted mean | 1e-6 |
| `--disable-link-graph` | Turn the link graph off (see [Ablations](#ablation-study)); **on by default** | off |

### Losses

$\mathcal{L} = \mathcal{L}_{\text{query}} + \alpha\,\mathcal{L}_{\text{hrta}} + \eta_p \mathcal{L}_{\text{proto}} + \eta_s \mathcal{L}_{\text{struct}} + \eta_{\text{div}} \mathcal{L}_{\text{div}} + \eta_{\text{pdiv}} \mathcal{L}_{\text{pdiv}}$

| Flag | Meaning | Default |
| --- | --- | --- |
| `--alpha` | $\alpha$: weight of $\mathcal{L}_{\text{hrta}}$ (as in RAA-KGC) | 0.2 |
| `--eta-proto` | $\eta_p$: weight of $\mathcal{L}_{\text{proto}}$ | 0.1 |
| `--eta-struct` | $\eta_s$: weight of $\mathcal{L}_{\text{struct}}$ | 0.1 |
| `--eta-div` | $\eta_{\text{div}}$: anchor-diversity loss (**off** by default) | 0.0 |
| `--eta-pdiv` | $\eta_{\text{pdiv}}$: prototype-specialisation loss (cosine between prototypes) | 0.01 |
| `--t` | Initial temperature of $\mathcal{L}_{\text{query}}$/$\mathcal{L}_{\text{hrta}}$ (learnable unless `--no-finetune-t`) | 0.05 |
| `--t-mem` | Initial temperature of $\mathcal{L}_{\text{proto}}$/$\mathcal{L}_{\text{struct}}$ (learnable unless `--no-finetune-t-mem`; inverse temperature clamped to $[1,100]$) | 0.05 |
| `--additive-margin` | Margin $\gamma$ on positives | 0.02 |
| `--use-self-negative` / `--no-use-self-negative` | Add the head entity as an extra negative for $S_q$ | on |

### Optimisation and training

| Flag | Meaning | Default |
| --- | --- | --- |
| `--lr` | LR of the two BERT encoders and the main temperature | 5e-5 |
| `--memory-lr` | LR of the small memory modules (anchor projection, prototypes, hop scorer, memory temperatures); they train from scratch. Our runs use 1e-4 | 1e-3 |
| `--wd` | Weight decay (memory modules use 0) | 1e-4 |
| `--epochs`, `-b/--batch-size` | | 20, 32 |
| `--lr-scheduler` | `linear` or `cosine` | linear |
| `--warmup` | Warm-up steps (capped to 10% of all steps) | 400 |
| `--grad-clip` | | 10 |
| `--pooling` | `cls`, `mean`, `max` | mean |
| `--use-amp` | Mixed precision | off |
| `--seed`, `-j/--workers`, `-p/--print-freq`, `--gpu` | | None, 4, 20, 0 |

### Model management and evaluation

| Flag | Meaning | Default |
| --- | --- | --- |
| `--model-dir` | Checkpoint directory | `data/<task>/checkpoint_arpm_v2` |
| `--eval-model-path` | Checkpoint used by evaluation | `<model-dir>/model_best.mdl` |
| `--checkpoint-metric` | Validation metric for `model_best.mdl` (`mrr`, `hit@1/3/10/50`) | mrr |
| `--full-eval-every-n-epochs` | Cadence of the full filtered-ranking validation | 1 |
| `--full-eval-batch-size` | Batch size of that validation pass | 8 |
| `--eval-every-n-step` | Mid-epoch checkpoint cadence (saves `model_last.mdl` only, no validation) | 0 |
| `--resume`, `--resume-path` | See [Resume](#resume) | off |
| `--rerank-n-hop`, `--neighbor-weight` | SimKGC-style graph re-ranking bonus, evaluation only (`--neighbor-weight 0` disables it) | 5, 0.05 |
| `--is-test` | Evaluation mode | off |

## Score coefficients

| Flag | Meaning | Default when unset |
| --- | --- | --- |
| `--coef-h`, `--coef-p`, `--coef-s` | $c_h, c_p, c_s$ | 1.0, 0.1, 0.1 |
| `--tune-coefs` | Grid search by average MRR on `--valid-path`, saved to `best_coefs.json` | off |
| `--coef-h-grid`, `--coef-p-grid`, `--coef-s-grid` | Search grids | `0 0.5 1 1.5` / `0 0.1 0.25 0.5 1` / `0 0.1 0.25 0.5 1` |

All grid points are scored from the same cached $S_q, S_{\text{hrta}}, S_p, S_{\text{struct}}$ matrices, so a large grid is cheap relative to the encoding pass. The tool warns when the best value sits at the upper edge of its grid. Always tune on the validation split and run the test split once afterwards. During training, the in-batch accuracy shown in the log and the per-epoch validation used for checkpoint selection use the default constants (or the `--coef-*` values you pass), **not** the tuned ones.

## Ablation study

Coefficient ablations are **evaluation-only** (no retraining); the others change the trained model.

| Ablation | Description | How |
| --- | --- | --- |
| RAA-KGC rule | Baseline inference from the same checkpoint | `--coef-h 1 --coef-p 0 --coef-s 0` |
| Query only | $S_q$ alone | `--coef-h 0 --coef-p 0 --coef-s 0` |
| No $S_p$ | Drop semantic memory | `--coef-p 0` together with explicit `--coef-h` and `--coef-s` |
| No $S_{\text{struct}}$ | Drop structural memory | `--coef-s 0` together with explicit `--coef-h` and `--coef-p` |
| A1 | Random (uniform) anchor weights instead of query-conditioned attention | `--random-anchor-selection` |
| A3 | Single prototype | `--num-prototypes 1` |
| A5 | Uniform hop weighting over non-empty hops | `--uniform-hop-weighting` |
| A6 | No global relation memory | `--global-budget 0` |
| A7 | No hop $\geq 1$ local anchors and no neighbour-text augmentation (hop-0 anchors are kept, since they need no graph traversal). Pair with `--neighbor-weight 0` for a graph-free evaluation | `--disable-link-graph` |
| Hop depth | Hop-0 only (no traversal) | `--num-hops 0` |
| No RAA path | Remove $S_{\text{hrta}}$ and $\mathcal{L}_{\text{hrta}}$ | `--anchor-num 0` |
| Diversity | Optional anchor-diversity regulariser (off by default) | `--eta-div 0.01` |
| No prototype specialisation | | `--eta-pdiv 0` |

Because unset coefficients fall back to `(1.0, 0.1, 0.1)`, always pass all three when you fix them.

## Checkpoints and model selection

Selection uses the **same filtered-ranking protocol as final evaluation** (forward and backward MRR/Hits@k against the whole entity dictionary), not an in-batch proxy. At the epoch boundary the trainer wraps the in-memory model as an `ARPMPredictor` (`ARPMPredictor.from_model`), encodes all entities, and calls `evaluate_predictor` (the same function as the standalone script, with `save_details=False`). The result of `--checkpoint-metric` decides whether `model_best.mdl` is overwritten.

- Every epoch writes `model_last.mdl` (full state: weights, optimizer, scheduler, scaler, args) for resuming.
- `model_best.mdl` (weights + args only) is written **only** after a validation pass that improves the metric. Epochs that skip validation never touch it.
- **If `--full-eval-every-n-epochs` is larger than `--epochs`, validation never runs and no `model_best.mdl` is produced**. Evaluate `model_last.mdl` instead (our WN18RR runs do this to save time).
- Only `model_last.mdl` and `model_best.mdl` are written; `--max-to-keep` therefore has no practical effect.

## Outputs

For an evaluation on `<split>` with checkpoint `<ckpt>`:

| File (next to the checkpoint) | Content |
| --- | --- |
| `best_coefs.json` | Tuned `(c_h, c_p, c_s)`, split tuned on, validation metrics |
| `task_hrt_<split>_<forward\|backward>_<ckpt>.json` | Per-query predictions: top-20 candidates with scores, rank, correctness |
| `task_hrt_<split>_<ckpt>.json` | Plain-text summary: coefficients and forward / backward / average metrics (despite the extension, not JSON) |
| `metrics_<split>_<ckpt>.json` | JSON summary written by `eval_wiki5m_trans.py` |

## Implementation notes

### Candidate retrieval and leakage control

Retrieval reads **only the training graph** (`dict_hub.get_train_triplet_dict`, independent of `--is-test`), so no validation or test label can enter a pool.

- During training the gold tail is excluded from hop-0 anchors, the query's own $(h,t)$ edge is hidden from the BFS (`get_hop_layers(exclude_edge=...)`), and the gold tail is removed from neighbour-text augmentation.
- During evaluation, sampling is deterministic per query (seeded by `head\trelation`).
- Hop indexing: hop-0 is "same head and relation" (no traversal), so `--num-hops N` gives $N+1$ slots. Global candidates and padding use `NO_HOP = -1`, and global sampling excludes pairs that share the query head (those belong to hop-0).
- Hop-0 anchors are never removed by `--anchor-budget`; only hop $\ge 1$ and global candidates are sub-sampled to fit the remaining room. `--local-per-hop-budget 0` yields exactly zero local anchors.

### Gradient flow (what trains what)

- The two BERT encoders are trained by $\mathcal{L}_{\text{query}}$ and $\mathcal{L}_{\text{hrta}}$ only, exactly as in RAA-KGC.
- The memory branch sees `q.detach()`, anchor embeddings $E_1(t_i)$ are computed under `no_grad` in eval mode, and the memory losses use `tail_vector.detach()`. $\mathcal{L}_{\text{proto}}$, $\mathcal{L}_{\text{struct}}$, $\mathcal{L}_{\text{div}}$, $\mathcal{L}_{\text{pdiv}}$ therefore update only the memory modules (anchor projection $W$, `ProtoGen`, `HopScorer`, memory temperatures).
- $\mathcal{L}_{\text{proto}}$ and $\mathcal{L}_{\text{struct}}$ are bidirectional in-batch cross-entropies that ignore queries with an empty memory (`_masked_bidirectional_ce`): $\mathcal{L}_{\text{struct}}$ additionally requires at least one local anchor.
- Temperature scaling, the additive margin and the self-negative column are training-only. Ranking always uses raw cosine scores.

### Memory construction

- **Anchor weights:** $\alpha = \mathrm{softmax}$ over valid slots of $a_i^\top W q/\tau_r$ ($W$ initialised to the identity).
- **Prototypes:** $u_{ik} = a_i^\top W_k q / T$ with $T$ = `--proto-attn-temperature`; $\rho_{ik} = \alpha_i e^{u_{ik}} / \sum_j \alpha_j e^{u_{jk}}$; $p_k = \sum_i \rho_{ik} a_i$. Each $W_k$ is initialised to the identity plus small noise, so every slot starts as a slightly different cosine attention. Prototypes are L2-normalised (zero stays zero).
- **Structural memory:** $m^{(\ell)} = \sum_i \alpha_i a_i / (\sum_i \alpha_i + \epsilon)$ over local anchors of hop $\ell$; $\beta = \mathrm{softmax}(\text{Linear}(q))$ masked so that **an empty hop gets exactly zero weight**; $m_{\text{struct}} = \sum_\ell \beta_\ell m^{(\ell)}$, L2-normalised. A query with no local anchor gets $m_{\text{struct}} = 0$ through an element-wise `torch.where`, so no NaN can propagate and $S_{\text{struct}} = 0$.
- **Zero-candidate queries:** the model never falls back to the query's own gold triple. All slots are padding, $\alpha$ and $\mathcal{L}_{\text{div}}$ are exactly zero, prototypes and $m_{\text{struct}}$ are zero vectors, and $S_p = S_{\text{struct}} = 0$ for that query.

### False-negative masking

The shared `triplet_mask` (other known true tails of the same $(h,r)$ appearing in the batch) is applied to $\mathcal{L}_{\text{query}}$, $\mathcal{L}_{\text{hrta}}$, $\mathcal{L}_{\text{proto}}$ and $\mathcal{L}_{\text{struct}}$. Pre-batch negatives are not implemented.

### Evaluation protocol

- Rank of the gold tail = 1 + number of entities with a strictly higher score, after filtering.
- Filtering (`evaluate.py::_filter_indices`) masks (a) every other known tail of $(h, r)$ over train+valid+test (including inverse triples), and (b) every entity $e$ that has the head as a tail of some triple, except the gold tail. Rule (b) is the same rule as in our `Baseline` (RAA-KGC) package; the `SimKGC_Restructured` and `HaSa_Restructured` packages apply only rule (a).
- The graph re-ranking bonus (`--neighbor-weight` on entities within `--rerank-n-hop` hops of the head) is added to $S_q$ only; it is additive, so it is equivalent to adding it to the total score.
- Both directions are evaluated (forward, and backward through `inverse <relation>`); reported numbers are their average.

### Link-graph semantics

The link graph is enabled by default. `--disable-link-graph` removes neighbour-text augmentation of entity descriptions and all hop $\ge 1$ anchors, but keeps hop-0 anchors (and hence $S_{\text{hrta}}$ and the hop-0 part of the memories). Graph re-ranking at evaluation builds its own graph, so set `--neighbor-weight 0` for a fully graph-free evaluation.

## Method-to-code mapping

| Method component | Implementation |
| --- | --- |
| Relation-aware candidate memory $\mathcal{A}_{\text{local}}, \mathcal{A}_{\text{global}}$, budget $M$ | `utils/candidate_pool.py::CandidatePoolBuilder`, `utils/triplet.py::LinkGraph.get_hop_layers` |
| Anchor-enhanced query (RAA path), $E_0(h,r,t_i)$ | `utils/doc.py::Example.vectorize(test=False)`, `model/models.py::ARPMModel.forward` |
| Query-conditioned anchor selection (RQ1) | `model/models.py::ARPMModel._memory` (`attn_proj`, $\alpha_i$) |
| Anchor diversity $\mathcal{L}_{\text{div}}$ | `model/modules.py::diversity_loss` |
| Multi-prototype semantic memory (RQ2) | `model/modules.py::ProtoGen`, `prototype_diversity` ($\mathcal{L}_{\text{pdiv}}$) |
| Adaptive structural memory (RQ3), empty-hop safeguards | `model/models.py::ARPMModel._structural_memory`, `model/modules.py::HopScorer` |
| Scores $S_q, S_p, S_{\text{struct}}$ and the combined score | `ARPMModel.score_query / score_prototypes / score_struct / combined_score` |
| Training objective | `model/trainer.py::Trainer._compute_losses` |
| Constant coefficients, grid search | `setting/config.py::get_coefs`, `evaluation/evaluate.py::resolve_coefs` |
| Filtered-ranking evaluation | `evaluation/evaluate.py::compute_metrics / evaluate_predictor` |

## Known limitations

- **Candidate-pool union.** $\mathcal{A}_{\text{local}} \cup \mathcal{A}_{\text{global}}$ is a concatenation, not a true set union. A triple reached at hop $\ell \ge 1$ can also be sampled into the global pool and appear twice under different tags. This only adds a little retrieval weight to that anchor.
- **Wikidata5M preprocessing.** In `preprocess/preprocess.py::preprocess_wiki5m` the line `partial(_process_line_wiki5m, id2ent=_load_wiki5m_id2ent)` passes the loader *function* instead of the `wiki5m_id2ent` dictionary; change it to `id2ent=wiki5m_id2ent` (as in `Baseline`) before preprocessing `wiki5m_trans` / `wiki5m_ind`.
- **DDP and long validation.** Under `torchrun`, only rank 0 runs the full validation while the other ranks wait at a barrier whose timeout is 60 minutes (`main.py`). Very large graphs may need a longer timeout or `--full-eval-every-n-epochs` set large.
- **No automated tests** are included. Empirical claims should be validated with full training runs on WN18RR / FB15k-237.

## Citation

ARPM-KGC builds on the baseline below; please cite it. A citation for ARPM-KGC itself will be added upon publication.

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

## License

No license is currently specified. Add one (for example MIT or Apache-2.0) before public release or redistribution.

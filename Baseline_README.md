# Baselines for RAA-KGC / ARPM-KGC

This document is the merged, cleaned-up reference for everything added to compare
against the proposed model: **12 baselines** total, split into four pieces.

```
Baselines/        7 triple-based methods: TransE, DistMult, ComplEx, RotatE, QuatE,
                  QIQE-KGC, ConvE, RGCN — one shared trainer/loss/evaluator
StAR/             text-based baseline (self-contained folder, mirrors SimKGC/)
HaSa/             text-based baseline (self-contained folder, mirrors SimKGC/)
SimKGC_patch/     2 files (config.py, trainer.py) that patch your existing SimKGC/
                  in place, adding MRR-based early stopping / checkpoint selection
```

All four come with **early stopping and best-checkpoint selection on validation MRR**
(filtered, forward+backward averaged — the same protocol `ARPM_KGC`/`Baseline`
already use), not just training loss or in-batch accuracy.

This README only covers **wn18rr** and **fb15k237**. All four pieces also support
`wiki5m_trans` / `wiki5m_ind` (see the `train_wiki.sh` / `eval_wiki5m_trans.*`
scripts and `--task` flags) but those aren't included here.

---

## 0. Prerequisites — preprocessed data

Nothing here re-implements preprocessing. All four pieces consume the **same**
preprocessed files your existing `Baseline/` or `SimKGC/` pipeline already produces
per task:

```
data/<task>/entities.json
data/<task>/train.txt.json
data/<task>/valid.txt.json
data/<task>/test.txt.json
```

Generate them once (either pipeline works, they use the same format):

```bash
python -m Baseline.preprocess.preprocess --task wn18rr
python -m Baseline.preprocess.preprocess --task fb15k237
# or, from inside SimKGC/:
./scripts/preprocess.sh WN18RR
./scripts/preprocess.sh FB15k237
```

**Data-directory naming gotcha:** `Baselines/` (the 7 embedding models) expects
**lowercase** directory names by default — `data/wn18rr/`, `data/fb15k237/`
(`Baselines/config.py`'s `--task` has `choices=['wn18rr', 'fb15k237', ...]` and
derives `--data-dir` from it directly). `SimKGC/`, `StAR/`, and `HaSa/`'s shell
scripts default to **mixed-case** directories instead — `data/WN18RR/`,
`data/FB15k237/` (see `train_wn.sh`/`train_fb.sh`'s `TASK=` variable). Both sets of
scripts accept a `DATA_DIR` env-var override, so the simplest fix is to keep one
preprocessed folder and point everything at it explicitly, e.g.:

```bash
DATA_DIR=./data/wn18rr    ./scripts/train_wn.sh
DATA_DIR=./data/fb15k237  ./scripts/train_fb.sh
```

---

## 1. `Baselines/` — TransE, DistMult, ComplEx, RotatE, QuatE, QIQE-KGC, ConvE, RGCN

Run from the **repo root** (the folder containing `Baselines/`, `Baseline/`, `SimKGC/`, etc.):

| `--model`  | Paper                              | Key flags |
|------------|-------------------------------------|-----------|
| `transe`   | Bordes et al. 2013                  | `--margin` (gamma), `--p-norm` (1 or 2) |
| `distmult` | Yang et al. 2015 (ICLR)             | — |
| `complex`  | Trouillon & Bouchard 2016 (ICML)    | `--weight-decay` (paper's L2 term; tuning away from 0 found up to +0.05 MRR) |
| `rotate`   | Sun et al. 2019 (ICLR)              | `--margin` (gamma), `--adv-temperature` |
| `quate`    | Zhang et al. 2019 (NeurIPS)         | plain QuatE — rotates head only |
| `qiqekgc`  | Li et al. 2023 (Information Sciences) | `--qiqe-alpha`, `--qiqe-beta`, `--qiqe-reg-weight` |
| `conve`    | Dettmers et al. 2018 (AAAI)         | `--conve-height`, `--conve-filters`, `--conve-kernel`, `--conve-*-dropout` |
| `rgcn`     | Schlichtkrull et al. 2018 (ESWC)    | `--rgcn-in-dim`, `--rgcn-hidden-dim`, `--rgcn-num-bases` |

> **On "QuatE":** the citation for that comparison row is actually Li et al. 2023's
> *quantum-embedding + quaternion-interaction* method, nicknamed QIQE-KGC in the
> paper itself — not the original Zhang et al. 2019 QuatE. Both are implemented:
> `--model qiqekgc` is the one that matches the citation; `--model quate` is plain
> QuatE (also used internally as one of QIQE-KGC's own baselines). QIQE-KGC's
> quaternion module (Eq. 9-16) and quantum score (Eq. 1) are implemented in full,
> as are its E2R-style regularizers (Eq. 23-26, 32); the logic-based `LLoss`
> (Eq. 27-31) is a documented omission — it needs T-box relation-logic annotations
> that standard KGC datasets don't ship with.

### Design notes
- **One shared loss** (RotatE's self-adversarial negative sampling) and **one
  shared trainer/dataset/evaluator** for all 7 models — only `score()`/`score_all()`
  differ per model (`Baselines/embedding_models/base_model.py`).
- Relations are inverse-augmented (`r_inv = r + num_relations`), so head-prediction
  is tail-prediction with the inverse relation — one code path, both directions.
- RGCN's graph encoder runs **once per epoch**, not once per batch
  (`common/trainer.py::_maybe_encode_graph`).
- Filtered ranking (MRR / Hits@1,3,10 / mean rank) matches the convention already
  used in `Baseline/evaluation/evaluate.py`.
- Best checkpoint = highest **validation** MRR, tracked via `EarlyStopping`
  (`--patience`, in number of *evals* set by `--eval-every`, not epochs).

### Train + test — WN18RR

```bash
# from repo root; each of these automatically evaluates model_best.mdl on the
# TEST split once training/early-stopping finishes, and writes
# data/wn18rr/checkpoint_<model>/test_metrics_<model>.json
python3 -m Baselines.main --model transe   --task wn18rr --embedding-dim 200 --margin 9   --neg-size 256
python3 -m Baselines.main --model rotate   --task wn18rr --embedding-dim 500 --margin 6   --adv-temperature 1.0
python3 -m Baselines.main --model distmult --task wn18rr --embedding-dim 200
python3 -m Baselines.main --model complex  --task wn18rr --embedding-dim 500 --weight-decay 0.05
python3 -m Baselines.main --model quate    --task wn18rr --embedding-dim 100
python3 -m Baselines.main --model qiqekgc  --task wn18rr --embedding-dim 100 --qiqe-alpha 0.5 --qiqe-beta 0.5
python3 -m Baselines.main --model conve    --task wn18rr --embedding-dim 200 --dropout 0.2 --conve-filters 32
python3 -m Baselines.main --model rgcn     --task wn18rr --embedding-dim 200 --rgcn-in-dim 200 --rgcn-hidden-dim 200
```

### Train + test — FB15k-237

```bash
python3 -m Baselines.main --model transe   --task fb15k237 --embedding-dim 200  --margin 9 --neg-size 128 --lr 5e-4
python3 -m Baselines.main --model rotate   --task fb15k237 --embedding-dim 1000 --margin 9 --adv-temperature 1.0 --neg-size 128 --lr 5e-4
python3 -m Baselines.main --model distmult --task fb15k237 --embedding-dim 200  --neg-size 128 --lr 5e-4
python3 -m Baselines.main --model complex  --task fb15k237 --embedding-dim 1000 --weight-decay 0.05 --neg-size 128 --lr 5e-4
python3 -m Baselines.main --model quate    --task fb15k237 --embedding-dim 100  --neg-size 128 --lr 5e-4
python3 -m Baselines.main --model qiqekgc  --task fb15k237 --embedding-dim 100  --qiqe-alpha 0.5 --qiqe-beta 0.5 --neg-size 128 --lr 5e-4
python3 -m Baselines.main --model conve    --task fb15k237 --embedding-dim 200  --dropout 0.2 --conve-filters 32 --neg-size 128 --lr 5e-4
python3 -m Baselines.main --model rgcn     --task fb15k237 --embedding-dim 200  --rgcn-in-dim 200 --rgcn-hidden-dim 200 --neg-size 128 --lr 5e-4
```

Or via the generic launcher: `Baselines/scripts/train.sh <model> <task> [extra args]`,
e.g. `Baselines/scripts/train.sh rotate fb15k237 --embedding-dim 1000 --margin 9`.

### Standalone test evaluation (no retraining)

Useful for re-checking a checkpoint, or evaluating on `valid` instead of `test`:

```bash
python3 -m Baselines.evaluate --model transe \
  --checkpoint data/wn18rr/checkpoint_transe/model_best.mdl \
  --data-dir data/wn18rr --split test

python3 -m Baselines.evaluate --model transe \
  --checkpoint data/fb15k237/checkpoint_transe/model_best.mdl \
  --data-dir data/fb15k237 --split test
```

> **Known gap:** `Baselines/evaluate.py`'s `--model` choices list omits `qiqekgc`
> (it's present in `factory.py` and `config.py`). Add `'qiqekgc'` to that
> `choices=[...]` list before using this standalone script for that baseline —
> training + the automatic post-training test evaluation in `Baselines/main.py`
> are unaffected, since that path doesn't use this restricted list.

### Suggested starting hyperparameters (not tuned — reasonable defaults)

- **WN18RR**: `--embedding-dim 200` (500 for RotatE/ComplEx), `--neg-size 256
  --lr 1e-3 --epochs 200 --patience 10`. TransE/RotatE: `--margin 6`. ConvE:
  `--dropout 0.2 --conve-filters 32`.
- **FB15k-237**: `--embedding-dim 200` (1000 for RotatE/ComplEx), `--neg-size 128
  --lr 5e-4 --epochs 200 --patience 10`. TransE/RotatE: `--margin 9`.

---

## 2. `SimKGC_patch/` — early stopping for your existing `SimKGC/`

Drop-in replacements for `SimKGC/config.py` and `SimKGC/trainer.py`; nothing else
in `SimKGC/` changes.

```bash
cp SimKGC_patch/config.py  SimKGC/config.py
cp SimKGC_patch/trainer.py SimKGC/trainer.py
```

What changed: `eval_epoch` (in-batch Acc@1) is kept for logging only. A new
`_compute_full_mrr()` reuses `evaluate.py::compute_metrics` to get real filtered
forward+backward MRR on `valid_path`; that MRR now drives early stopping
(`--early-stop-patience`, default 5) and which checkpoint becomes
`model_best.mdl`. New flags: `--early-stop-patience`, `--full-eval-every-n-epoch`
(default 1 — raise it on wiki5m, expensive full-corpus MRR there),
`--mrr-eval-batch-size`.

### Train — WN18RR / FB15k-237

```bash
cd SimKGC
./scripts/train_wn.sh          # bert-base-uncased, lr 5e-5, batch 1024, 50 epochs
./scripts/train_fb.sh          # bert-base-uncased, lr 1e-5, batch 1024, 10 epochs
```

Checkpoints (best by full validation MRR, with early stopping once the patch is
applied) land in `checkpoint/WN18RR_<timestamp>/model_best.mdl` and
`checkpoint/FB15k237_<timestamp>/model_best.mdl`.

### Evaluate on the test set

```bash
cd SimKGC
./scripts/eval.sh checkpoint/WN18RR_<timestamp>/model_best.mdl   WN18RR
./scripts/eval.sh checkpoint/FB15k237_<timestamp>/model_best.mdl FB15k237
```

`eval.sh` sets `--is-test` and points `--valid-path` at `test.txt.json`, so this
runs the filtered forward+backward protocol on the **test** split and writes
`metrics_test.txt.json_model_best.mdl.json` plus per-direction predictions
(`eval_test.txt.json_{forward,backward}_model_best.mdl.json`) next to the
checkpoint. `WN18RR` uses `--rerank-n-hop 5` (sparser graph); `FB15k237` and
`wiki5m_ind` use `--neighbor-weight 0.0` / defaults as set inside the script.

`StAR/` and `HaSa/` (§3, §4) reuse this exact `eval.sh` unchanged, so the same
command shape applies there too.

### Bug fixed in all three copies (StAR / HaSa / SimKGC_patch)

Your original `SimKGC/config.py` has `--is-test` defined as
`default=True, action='store_true'` — a `store_true` flag can never be turned
*off* from the command line, so with `default=True` it can never be
`False`. Since eval scripts pass `--is-test` explicitly and training scripts never
do, the default should be `False`. Fixed to `default=False` in `StAR/config.py`,
`HaSa/config.py`, and `SimKGC_patch/config.py`. As shipped, this silently
disabled in-batch false-negative masking during training (a quality issue, not a
crash) and would have crashed StAR's negative sampler outright. Apply the same
one-line fix to your actual `SimKGC/config.py` if you'd like (not touched, since
it wasn't part of the patch).

---

## 3. `StAR/` — Structure-Augmented Text Representation (Wang et al., WWW'21)

Siamese (weight-tied) BERT encoder: `u = Pool(Enc([h;r]))`, `v = Pool(Enc([t]))`.
Trained with the paper's two actual objectives over **explicit per-triple
negatives** (`K=5` by default, corrupting head or tail uniformly — not in-batch
contrastive negatives like SimKGC):
- `L^c`: BCE classification over `c = [u; u*v; u-v; v]` (Eq. 8-9, 12)
- `L^d`: margin hinge loss on `s^d = -||u-v||_2` (Eq. 11, 13)
- `L = L^c + gamma * L^d` (Eq. 14, `--structure-loss-weight`)

Two documented deviations: (1) ranking at inference uses `s^d`, not the paper's
`s^c` — rank-equivalent to this repo's shared dot-product evaluator for
L2-normalized vectors, and near-identical in the paper's own ablation (Table 7:
Hits@10 .701 vs .709, MRR .406 vs .401 for `s^d` alone); (2) the §3.4
self-adaptive RotatE ensemble isn't implemented.

`doc.py`, `dict_hub.py`, `predict.py`, `evaluate.py`, `metric.py`, `rerank.py` are
unchanged from `SimKGC/`; `star_data.py` (negative sampling) and
`models.py`/`trainer.py` (the two objectives above) are new.

### Train — WN18RR / FB15k-237

```bash
cd StAR
./scripts/train_wn.sh          # RoBERTa-base, lr 1e-5, batch 16, 7 epochs
./scripts/train_fb.sh          # RoBERTa-base, lr 1e-5, batch 16, 7 epochs
```

Checkpoints (best by full validation MRR, with early stopping) land in
`checkpoint/WN18RR_<timestamp>/model_best.mdl` and
`checkpoint/FB15k237_<timestamp>/model_best.mdl`.

### Evaluate on the test set

```bash
cd StAR
./scripts/eval.sh checkpoint/WN18RR_<timestamp>/model_best.mdl   WN18RR
./scripts/eval.sh checkpoint/FB15k237_<timestamp>/model_best.mdl FB15k237
```

Runs the filtered forward+backward protocol on `test.txt.json` and writes
`metrics_test.txt.json_model_best.mdl.json` plus per-example predictions
(`eval_test.txt.json_{forward,backward}_model_best.mdl.json`) next to the
checkpoint.

---

## 4. `HaSa/` — Hardness- and Structure-Aware Contrastive KGE (Zhang, Zhang & Molybog, WWW'24)

A genuinely different architecture: **one shared encoder** applied separately to
head, relation, and tail text (not head+relation concatenation), aggregated via a
GRU into the query embedding `e_hr` (Section 3). The loss implements Algorithm 1 /
Eq. 6-9 in `trainer.py::_hasa_loss`: in-batch hardness-weighted negatives plus a
false-negative correction term estimated by sampling the head entity's ≤2-hop
link-graph neighbourhood (`alpha(t|e_hr)`, Eq. 9).

Documented deviations: (a) added a learnable inverse-temperature (`log_inv_t`),
since the paper's raw `exp(e_hr·e_t)` on bounded `[-1,1]` dot products gives a
very flat softmax otherwise; (b) negative count `K` is simplified to "all other
tails in the batch" rather than the paper's exact `K=5|T_batch|-1` bookkeeping
(Eq. 3 confirms the negative distribution's support is the same either way);
(c) HaSa+'s (Section 6) extra negative-query loss term isn't implemented.

`doc.py` is modified from `SimKGC/` (adds standalone relation-only tokenization,
since head/relation/tail are three separate texts); `triplet_mask.py` was removed
as dead code (HaSa's loss doesn't use in-batch triplet masking);
`models.py`/`trainer.py` are new. `dict_hub.py`, `predict.py`, `evaluate.py`,
`rerank.py` are unchanged from `SimKGC/`.

### Train — WN18RR / FB15k-237

```bash
cd HaSa
./scripts/train_wn.sh          # BERT-base, embedding-dim 500, tau 2e-5, 10 epochs
./scripts/train_fb.sh          # BERT-base, embedding-dim 500, tau 1e-4, 10 epochs
```

`--tau` (`p(l=fact|e_hr)`, Section 7.5) is dataset-specific: the paper's best
values are `2e-5` on WN18RR and `1e-4` on FB15k-237 — already set in the two
scripts above.

### Evaluate on the test set

```bash
cd HaSa
./scripts/eval.sh checkpoint/WN18RR_<timestamp>/model_best.mdl   WN18RR
./scripts/eval.sh checkpoint/FB15k237_<timestamp>/model_best.mdl FB15k237
```

Same output convention as StAR/SimKGC (filtered forward+backward MRR/Hits@k on
`test.txt.json`, plus per-example prediction dumps).

---

## Notes / things worth double-checking together

- **RGCN at wiki5m scale**: pure PyTorch, loops over relation types once per
  epoch — fine on wn18rr/fb15k237, will be slow at wiki5m's ~20M-edge scale.
- All 7 `Baselines/` models score the *entire* entity set per query, chunked via
  `--eval-entity-chunk` to bound memory.
- None of these four pieces have been run end-to-end in this environment (no GPU /
  no install budget here) — treat this as a careful implementation + review pass
  rather than a benchmarked one. Flag anything that doesn't match your own runs.
- `ConvE`'s three dropout rates (embedding/feature-map/projection) are tuned
  independently per the paper (best found 0.2/0.2/0.3) via `--conve-input-dropout`,
  `--conve-feature-dropout`, `--conve-hidden-dropout` — don't collapse them back
  to one shared `--dropout`.
- `RGCN`'s degree normalization uses **total** in-degree across all relations
  (`common/data.py::build_message_passing_graph`), matching the paper's own
  recommendation specifically for the link-prediction setting — not per-relation
  in-degree.

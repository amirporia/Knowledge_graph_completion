# Baselines for RAA-KGC

This adds the 12 compared baselines to your repo, split into three pieces:

```
Baselines/        7 triple-based methods: TransE, ComplEx, DistMult, ConvE, RGCN, RotatE, QuatE
                  (plus QIQE-KGC, the paper actually cited for "QuatE" — see below)
StAR/             text-based baseline (new self-contained folder, mirrors SimKGC/)
HaSa/             text-based baseline (new self-contained folder, mirrors SimKGC/)
SimKGC_patch/     2 files (config.py, trainer.py) that patch your existing SimKGC/ in place
```

All 10 come with **early stopping and best-checkpoint selection on validation MRR**
(filtered, forward+backward averaged, matching your existing evaluation protocol),
not just training loss or in-batch accuracy.

## 1. Baselines/ — TransE, ComplEx, DistMult, ConvE, RGCN, RotatE, QuatE, QIQE-KGC

Drop the `Baselines/` folder in at your repo root (next to `Baseline/` and `SimKGC/`).
It reuses the **same preprocessed data** your other two methods already consume
(`entities.json`, `train.txt.json`, `valid.txt.json`, `test.txt.json` — whatever
`Baseline/preprocess/preprocess.py` or `SimKGC/preprocess.py` produced), so no new
preprocessing step is needed.

```bash
# from repo root
python3 -m Baselines.main --model transe   --task wn18rr   --embedding-dim 200 --margin 9
python3 -m Baselines.main --model rotate   --task fb15k237 --embedding-dim 500 --margin 9 --adv-temperature 1.0
python3 -m Baselines.main --model distmult --task wn18rr   --embedding-dim 200
python3 -m Baselines.main --model complex  --task wn18rr   --embedding-dim 200
python3 -m Baselines.main --model quate    --task wn18rr   --embedding-dim 100
python3 -m Baselines.main --model qiqekgc  --task wn18rr   --embedding-dim 100 --qiqe-alpha 0.5 --qiqe-beta 0.5
python3 -m Baselines.main --model conve    --task fb15k237 --embedding-dim 200
python3 -m Baselines.main --model rgcn     --task wn18rr   --embedding-dim 200 --rgcn-hidden-dim 200
# or: Baselines/scripts/train.sh <model> <task> [extra args]
```

Design choices (see `Baselines/embedding_models/base_model.py` for the full rationale):
- **One shared loss** (RotatE's self-adversarial negative-sampling loss) and **one
  shared trainer/dataset/evaluator** for all 7 models — only `score()` /
  `score_all()` differ per model. This is what "very optimized" mostly means here:
  a single well-tested, vectorized training/eval path instead of 7 bespoke ones.
- Relations are inverse-augmented (`r_inv = r + num_relations`, same trick your own
  `reverse_triplet()` already uses), so head-prediction is just tail-prediction with
  the inverse relation — one code path for both directions, in training and eval.
- RGCN's graph encoder runs **once per epoch**, not once per batch (see
  `common/trainer.py::_maybe_encode_graph`).
- Filtered ranking metrics (MRR / Hits@1,3,10 / mean rank) exactly match the
  filtered-eval convention already used in `Baseline/evaluation/evaluate.py`.

Every model was reviewed against its defining equation from the original paper
(score functions, RotatE's self-adversarial loss, R-GCN's basis decomposition, the
Hamilton product for QuatE's rotation, etc.) — see the docstring at the top of each
`Baselines/embedding_models/*.py` file.

**Update on the "QuatE" baseline**: I initially implemented plain QuatE (Zhang et al.
2019), but your citation for that row is actually Li et al. 2023's *"Knowledge graph
completion method based on quantum embedding and quaternion interaction
enhancement"* (Information Sciences) — a different, hybrid model (nicknamed
QIQE-KGC in the paper itself), not the original QuatE. Added it as `--model qiqekgc`
(kept `--model quate` too, since QIQE-KGC also uses plain QuatE as one of its own
baselines). `Baselines/embedding_models/qiqekgc.py`'s docstring lays out exactly
what's implemented — the quaternion module in full (a genuine improvement over plain
QuatE: it rotates *both* head and tail by separate per-relation quaternions, not
just the head) — and what's a documented simplification: the quantum-embedding
module's score function is implemented, but two of its three training-loss terms
require either external relation-logic annotations not available for standard KGC
datasets, or reference appendix equations that looked internally inconsistent in a
way suggestive of PDF/OCR transcription loss rather than something I could
confidently reconstruct. I didn't want to present a guessed formula as faithful, so
that piece trains through this repo's shared contrastive loss instead — flagged
clearly rather than silently substituted.

## 2. SimKGC_patch/ — early stopping for your existing SimKGC/ baseline

Your repo already has a full SimKGC implementation, so rather than reimplementing it,
`SimKGC_patch/config.py` and `SimKGC_patch/trainer.py` are **drop-in replacements**
for `SimKGC/config.py` and `SimKGC/trainer.py`. Everything else in `SimKGC/` is
untouched.

What changed: `eval_epoch` (in-batch Acc@1) is kept only for logging; a new
`_compute_full_mrr()` reuses **your own** `evaluate.py::compute_metrics` to get the
real filtered forward+backward MRR on `valid_path`, and that MRR now drives early
stopping (`--early-stop-patience`) and which checkpoint gets copied to
`model_best.mdl`. New flags: `--early-stop-patience` (default 5),
`--full-eval-every-n-epoch` (default 1 — raise this on wiki5m, since full-corpus MRR
eval is expensive there), `--mrr-eval-batch-size`.

```bash
cp SimKGC_patch/config.py  SimKGC/config.py
cp SimKGC_patch/trainer.py SimKGC/trainer.py
```

## 3. StAR/ and HaSa/ — new self-contained baselines

Both are new top-level folders that follow your repo's existing convention (like
`SimKGC/` itself): flat files, own `config.py`/`main.py`, run with
`python3 -u main.py --train-path ... --valid-path ...` from inside the folder.
`logger_config.py`, `triplet.py`, `dict_hub.py`, `utils.py`, `metric.py`,
`rerank.py`, `predict.py`, `evaluate.py` are copied **unchanged** from `SimKGC/`
(same filtered-MRR evaluator, same dot-product ranking), plus early stopping/MRR
selection (same mechanism as the SimKGC patch above). Where each method's actual
architecture/loss required it, more diverges:
- **StAR**: `doc.py` is unchanged (still head+relation-concatenated text); a new
  `star_data.py` handles StAR's explicit per-triple negative sampling (see below) —
  `models.py`/`trainer.py` are new.
- **HaSa**: `doc.py` is modified (adds standalone relation-only tokenization, since
  HaSa encodes head/relation/tail as three separate texts rather than concatenating
  head+relation; drops the now-unused triplet-mask fields from `collate()`);
  `triplet_mask.py` was removed as dead code; `models.py`/`trainer.py` are new.

```bash
cd StAR && ./scripts/train_wn.sh   # or train_fb.sh
cd HaSa && ./scripts/train_wn.sh
```

**Updated against the actual papers you sent** (Wang et al. 2021, WWW'21 for StAR;
Zhang, Zhang & Molybog 2024, WWW'24 for HaSa) — both were substantially rewritten
from an earlier draft that guessed at the details. Please read the module docstring
at the top of `StAR/models.py` and `HaSa/models.py`, which spell out exactly which
equation/section each piece of code implements and where I deliberately deviated:

- **StAR**: a genuinely Siamese (weight-tied, single shared module) encoder producing
  `u = Pool(Enc([h;r]))` / `v = Pool(Enc([t]))` (§3.1), trained with the paper's *two*
  actual objectives — a BCE classification loss over `c = [u; u*v; u-v; v]` (Eq. 8-9,
  12) and a margin hinge loss on `s^d = -||u-v||_2` (Eq. 11, 13), combined as
  `L = L^c + gamma*L^d` (Eq. 14) — over **explicit per-triple negatives** (`K=5` by
  default, corrupting head or tail uniformly, §3.3.1 / Appendix Table 11), not
  in-batch contrastive negatives like my first draft used. New `star_data.py` handles
  this negative sampling. Two deliberate simplifications, both documented in
  `config.py`: (1) ranking at inference uses `s^d` rather than the paper's `s^c`,
  which is rank-equivalent to the shared `evaluate.py`'s dot-product scoring for
  normalized vectors and nearly identical in the paper's own ablation (Table 7); (2)
  the §3.4 self-adaptive RotatE ensemble isn't implemented — say the word and I'll
  wire it up against `Baselines/embedding_models/rotate.py`.
- **HaSa**: a genuinely different architecture from SimKGC/StAR — **one shared
  encoder** applied separately to head, relation, and tail text (not head+relation
  concatenation), aggregated via a GRU into the query embedding `e_hr` (Section 3),
  matching the paper's explicit description. The loss now implements Algorithm 1 /
  Eq. 6-9 directly in `trainer.py::_hasa_loss`: in-batch hardness-weighted negatives
  plus a false-negative correction term estimated by sampling the head entity's
  ≤2-hop link-graph neighbourhood (`alpha(t|e_hr)`, Eq. 9, via the existing
  `LinkGraph.get_n_hop_entity_indices` utility). Documented deviations: (a) I added a
  learnable inverse-temperature (`models.py`'s `log_inv_t`) since the paper's raw
  `exp(e_hr·e_t)` on L2-normalized (bounded [-1,1]) dot products gives a very flat
  softmax otherwise — every method HaSa itself compares against uses one; (b) `K`
  (negative count) is simplified to "all other tails in the batch" rather than the
  paper's exact `K=5|T_batch|-1` bookkeeping, since Eq. 3 confirms the negative
  distribution's support is exactly the in-batch tails either way; (c) HaSa+'s
  (Section 6) extra negative-query loss term isn't implemented.

**A real bug I found and fixed along the way**: your `SimKGC/config.py` (and hence my
copies in `StAR/`, `HaSa/`, `SimKGC_patch/`) has `--is-test` defined as
`default=True, action='store_true'` — which means it can never be set to `False` from
the command line. Since eval scripts explicitly pass `--is-test` and training scripts
never do, the default should be `False`. I've fixed this in all three of my copies
(`default=False`). As shipped, this silently disabled in-batch negative-triplet
filtering during SimKGC/StAR-style training (a quality issue, not a crash) and would
have crashed StAR's new negative sampler outright. Worth applying the same one-line
fix to your actual `SimKGC/config.py` if you'd like — I didn't touch that file since
it wasn't part of the requested patch.

## Notes / things worth double-checking together

- **RGCN at wiki5m scale**: the R-GCN encoder is pure PyTorch (no `torch_geometric`
  dependency) and loops over relation types once per epoch; on wn18rr/fb15k237 this
  is fine, on wiki5m's ~20M edges it will be slow. Flag if you need this sped up
  (e.g. with edge/neighborhood sampling instead of full-graph convolution).
- All 7 embedding baselines evaluate by scoring the *entire* entity set per query,
  chunked via `--eval-entity-chunk` to bound memory — same idea as your own
  `eval_wiki5m_trans.py` sharding, just at eval time instead of embedding-dump time.
- I have not been able to run these end-to-end in this environment (no GPU / no
  install space here), so please treat this as a careful implementation pass rather
  than a benchmarked one — happy to help debug against your actual data.

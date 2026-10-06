# HaSa_Restructured

HaSa re-implemented with the same layout, data format, CLI and evaluation as `SimKGC_Restructured`.
`--method` selects what is trained; everything else (preprocessing, dataloader, checkpoints, filtered
forward+backward link-prediction evaluation) is shared, so baselines and HaSa are directly comparable.

| `--method`         | model / loss                                                         |
|--------------------|----------------------------------------------------------------------|
| `simkgc` (default) | SimKGC baseline, code path identical to `SimKGC_Restructured`        |
| `hasa`             | HaSa: hard + false negatives, debiased hardness-aware InfoNCE        |
| `hasa_hard_bias`   | Hard InfoNCE (ablation)                                              |
| `hasa_wohard_bias` | simple InfoNCE with random extra negatives (ablation)                |

Add `--plus` for HaSa+.

## Usage (run from the folder containing `HaSa_Restructured/` and `data/`)
```
bash HaSa_Restructured/scripts/preprocess.sh wn18rr          # same data files as SimKGC
bash HaSa_Restructured/scripts/train_wn.sh                   # SimKGC baseline
bash HaSa_Restructured/scripts/train_hasa_wn.sh              # HaSa  (METHOD=hasa_hard_bias ... for ablations)
bash HaSa_Restructured/scripts/eval.sh      <ckpt> wn18rr    # SimKGC (graph re-rank on)
bash HaSa_Restructured/scripts/eval_hasa.sh <ckpt> wn18rr    # HaSa   (graph re-rank off)
```
The method is stored in the checkpoint, so evaluation needs no `--method`.

## Layout
`model/models.py` (SimKGC + HaSa encoders), `model/losses.py` (HaSa losses), `model/negative_sampler.py`
(embedding bank, hard / false negatives), `model/trainer.py` (one loop for all methods),
`utils/doc.py` (adds relation tokens for HaSa, `--entity-text`), `evaluation/*` (metric-compatible).

## Differences from the original HaSa code (deliberate)
* Entities are keyed by id instead of name; evaluation is the full SimKGC protocol (all test triples,
  both directions, filtered with train+valid+test) instead of 12 batches of the test set.
* Loss is rewritten in a numerically stable form (verified equal to the original formula).
* Original `hard_tails[i:i+hard]` was misaligned with the flat hard-negative list; it is aligned here
  (each sample's own hard negatives are tested against its 2-hop neighbourhood).
* The original uses the embedding dim (500) as N in the debiased loss (`_, num_neg = v_h_t.shape`); kept by
  default (`--neg-count-mode dim`), `count` uses the real number of negatives.
* Non-finite loss/gradients skip the step (original `continue`), DDP-safe.
* HaSa is not supported on wiki5m (needs an embedding bank of all entities).
* Defaults follow the original: shared encoder `all-mpnet-base-v2`, `pooler_output`, no L2 norm, AdamW wd 0.01,
  clip 5, seed 40. Inputs use the SimKGC text (`name: description`, 50 tokens) for both methods; use
  `--entity-text name` for name-only inputs.

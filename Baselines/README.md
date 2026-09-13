# Baselines/ — triple-based KGE methods

7 methods, 1 shared trainer/loss/evaluator (see `embedding_models/base_model.py` for why).
Run from the repo root: `python3 -m Baselines.main --model <name> --task <task> [flags]`.

| `--model`  | Paper                              | Key flags |
|------------|-------------------------------------|-----------|
| `transe`   | Bordes et al. 2013                  | `--margin` (gamma), `--p-norm` (1 or 2) |
| `distmult` | Yang et al. 2015 (ICLR)             | — |
| `complex`  | Trouillon & Bouchard 2016 (ICML)    | `--weight-decay` (paper's L2 term; tuning away from 0 found up to +0.05 MRR) |
| `rotate`   | Sun et al. 2019 (ICLR)              | `--margin` (gamma), `--adv-temperature` |
| `quate`    | Zhang et al. 2019 (NeurIPS)         | — (plain QuatE; rotates head only) |
| `qiqekgc`  | **Li et al. 2023 (Info. Sci.)**     | `--qiqe-alpha`, `--qiqe-beta`, `--qiqe-reg-weight` — **this is the paper actually cited for the "QuatE" baseline in the original request**; see `qiqekgc.py`'s module docstring for exactly what's implemented vs. simplified (its quantum-embedding module's logic/membership losses require either external relation-logic annotations or ambiguous appendix equations, so those two pieces are documented omissions, not silent gaps) |
| `conve`    | Dettmers et al. 2018 (AAAI)         | `--conve-height`, `--conve-filters`, `--conve-kernel`, `--conve-input-dropout`, `--conve-feature-dropout`, `--conve-hidden-dropout` |
| `rgcn`     | Schlichtkrull et al. 2018 (ESWC)    | `--rgcn-in-dim`, `--rgcn-hidden-dim`, `--rgcn-num-bases` |

**Verified against the original papers** (ConvE, RotatE, R-GCN, ComplEx, and
QIQE-KGC's cited E2R source were all checked directly): ComplEx's score function
matches Eq. 11 exactly (confirmed both symbolically and numerically — see
`complex.py`'s docstring for the derivation, since the two formulas look different
at first glance despite being algebraically identical); RotatE's score/loss/
initialization match the paper exactly; ConvE's architecture matched already, but
its three independently-tuned dropout rates (embedding/feature-map/projection) were
previously collapsed into one shared `--dropout` — now three separate flags
matching the paper's best setting (0.2/0.2/0.3); RGCN's message-passing and basis
decomposition matched, but its normalization constant used per-relation in-degree
where the paper specifically recommends total in-degree across all relations for
*link prediction* — fixed in `common/data.py`. See each model's module docstring
for the full verification notes, including what's still a documented simplification
(RGCN's block-decomposition alternative and edge dropout; QIQE-KGC's relation-logic
loss, which needs T-box annotations no standard KGC dataset ships with).

Shared flags that matter most: `--embedding-dim`, `--neg-size` (negatives per positive,
default 256), `--batch-size`, `--lr`, `--epochs`, `--patience` (early stopping, in
number of *evals*, not epochs — see `--eval-every`), `--eval-entity-chunk` (memory vs.
speed at eval time).

Best checkpoint (by validation MRR) is written to `<model-dir>/model_best.mdl`;
`Baselines/main.py` automatically reloads it and reports test-set metrics once
training stops. To evaluate a checkpoint later without retraining:

```bash
python3 -m Baselines.evaluate --model transe \
  --checkpoint data/wn18rr/checkpoint_transe/model_best.mdl \
  --data-dir data/wn18rr --split test
```

## Suggested starting hyperparameters (not tuned — a reasonable starting point)

- **WN18RR**: `--embedding-dim 200`(500 for RotatE/ComplEx) `--neg-size 256 --lr 1e-3 --epochs 200 --patience 10`
  TransE/RotatE: `--margin 6`. ConvE: `--dropout 0.2 --conve-filters 32`.
- **FB15k237**: `--embedding-dim 200`(1000 for RotatE/ComplEx) `--neg-size 128 --lr 5e-4 --epochs 200 --patience 10`
  TransE/RotatE: `--margin 9`.
- **wiki5m_trans / wiki5m_ind**: start much smaller-scale than the above (large
  `--eval-entity-chunk` values will OOM); expect RGCN in particular to need real
  tuning/optimization work at this scale (see top-level README).

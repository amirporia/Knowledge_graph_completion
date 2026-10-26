# Knowledge Graph Completion: ARPM-KGC and Reproduced Baselines

Code accompanying our paper on **ARPM-KGC** (Adaptive Relation-Aware Prototype Memory for Knowledge Graph Completion), together with clean re-implementations of the baselines it is compared against. All four packages share the same data format, preprocessing, text-based dual-encoder design and filtered link-prediction evaluation (MRR, Hits@1/3/10/50, mean rank on WN18RR, FB15k-237 and Wikidata5M).

## Contents

- [Repository layout](#repository-layout)
- [Methods](#methods)
- [Installation](#installation)
- [Data](#data)
- [Quick start](#quick-start)
- [Comparing methods fairly](#comparing-methods-fairly)
- [Results](#results)
- [Citation and acknowledgements](#citation-and-acknowledgements)

## Repository layout

```
Knowledge_graph_completion/
  data/                              # you create this (see "Data")
  Baseline/                          # RAA-KGC (relation-aware anchor enhancement)
  SimKGC_Restructured/               # SimKGC
  HaSa_Restructured/                 # HaSa (+ SimKGC code path via --method simkgc)
  ARPM_KGC_DataParallel_original/    # ARPM-KGC (proposed); has its own README
```

Every package is a Python package run **from the repository root** with `python -m <Package>.<module>` (never `cd` into a package). All packages read and write `data/<task>/`.

## Methods

| Package | Method | Idea | Details |
| --- | --- | --- | --- |
| `SimKGC_Restructured` | SimKGC (Wang et al., ACL 2022) | Two BERT encoders, InfoNCE with in-batch, self and optional pre-batch negatives | `scripts/` |
| `HaSa_Restructured` | HaSa | Hardness-aware, debiased InfoNCE with mined hard and likely-false negatives. `--method` selects `simkgc`, `hasa`, `hasa_hard_bias`, `hasa_wohard_bias`; `--plus` gives HaSa+. Not supported on Wikidata5M | [`HaSa_Restructured/README.md`](HaSa_Restructured/README.md) |
| `Baseline` | RAA-KGC (Yuan et al., AAAI 2025) | SimKGC-style dual encoder plus relation-aware anchors; loss $\alpha\mathcal{L}_{hrta}+\mathcal{L}_{hr}$, inference score $\cos(e_{hr},e_t)+\cos(e^{avg}_{hrta},e_t)$ | `Baseline/` |
| `ARPM_KGC_DataParallel_original` | **ARPM-KGC (ours)** | RAA-KGC encoders plus query-conditioned anchor selection, multi-prototype semantic memory and hop-aware structural memory; score $S_q + c_h S_{hrta} + c_p S_p + c_s S_{struct}$ with coefficients tuned on the validation split | [`ARPM_KGC_DataParallel_original/README.md`](ARPM_KGC_DataParallel_original/README.md) |

With $c_h=1,\ c_p=c_s=0$, ARPM-KGC reproduces the RAA-KGC inference rule exactly.

## Installation

```bash
git clone https://github.com/amirporia/Knowledge_graph_completion.git
cd Knowledge_graph_completion
pip install torch transformers tqdm        # PyTorch 2.x; 2.3+ recommended
```

A CUDA GPU is strongly recommended. All training entry points support single-GPU, `nn.DataParallel` (plain `python` on a multi-GPU machine) and DDP (`torchrun --nproc_per_node=N -m <Package>.main ...`; `--batch-size` is then per process).

## Data

Create `data/<task>/` at the repository root and place the raw files there:

| Task | Required raw files |
| --- | --- |
| `wn18rr` | `train.txt`, `valid.txt`, `test.txt`, `wordnet-mlj12-definitions.txt` |
| `fb15k237` | `train.txt`, `valid.txt`, `test.txt`, `FB15k_mid2description.txt`, `FB15k_mid2name.txt` |
| `wiki5m_trans`, `wiki5m_ind` | run `bash SimKGC_Restructured/scripts/download_wikidata5m.sh` (or the identical script under `HaSa_Restructured/scripts/`) |

Preprocessing converts the raw files to `*.txt.json`, `entities.json` and `relations.json` in the same folder:

```bash
bash SimKGC_Restructured/scripts/preprocess.sh wn18rr                              # or HaSa_Restructured/scripts/preprocess.sh
python -m ARPM_KGC_DataParallel_original.preprocess.preprocess --task wn18rr       # equivalent, for Baseline / ARPM
python -m Baseline.preprocess.preprocess --task wn18rr                             # equivalent
```

Run it once per task; the output can be shared by all packages.

## Quick start

All examples use WN18RR; replace `wn18rr` with `fb15k237` etc. Checkpoints go to `data/<task>/checkpoint*` unless `--model-dir` is given.

### SimKGC

```bash
bash SimKGC_Restructured/scripts/train_wn.sh                      # also train_fb.sh, train_wiki.sh
bash SimKGC_Restructured/scripts/eval.sh <path/to/model_best.mdl> wn18rr
```

### HaSa

```bash
bash HaSa_Restructured/scripts/train_hasa_wn.sh                   # METHOD=hasa_hard_bias ... for ablations
bash HaSa_Restructured/scripts/eval_hasa.sh <ckpt> wn18rr         # graph re-ranking off for HaSa
bash HaSa_Restructured/scripts/train_wn.sh                        # SimKGC baseline inside the HaSa package
```

### RAA-KGC (Baseline)

```bash
python -m Baseline.main --task wn18rr --model-dir data/wn18rr/checkpoint_raa \
  --pretrained-model bert-base-uncased --pooling mean --max-num-tokens 50 \
  --epochs 20 --batch-size 32 --lr 5e-5 --wd 1e-4 --warmup 400 --grad-clip 10 \
  --t 0.05 --additive-margin 0.02 --alpha 0.2 --anchor-num 4 \
  --use-link-graph --use-amp --workers 4 --seed 42

python -m Baseline.evaluation.evaluate --task wn18rr \
  --eval-model-path data/wn18rr/checkpoint_raa/model_best.mdl \
  --pretrained-model bert-base-uncased --pooling mean --max-num-tokens 50 --batch-size 32 \
  --is-test --valid-path data/wn18rr/test.txt.json --rerank-n-hop 2 --neighbor-weight 0.05
```

Note that `--use-link-graph` is **opt-in** in `Baseline`.

### ARPM-KGC

```bash
python -m ARPM_KGC_DataParallel_original.main --task wn18rr --model-dir data/wn18rr/checkpoint_arpm \
  --epochs 20 --batch-size 32 --lr 5e-5 --memory-lr 1e-4 --anchor-num 4 --num-hops 2 \
  --num-prototypes 4 --local-per-hop-budget 5 --global-budget 10 --anchor-budget 25 \
  --use-amp --seed 42
```

Tuning the three score coefficients on validation, evaluating on test, all flags and ablations are documented in [`ARPM_KGC_DataParallel_original/README.md`](ARPM_KGC_DataParallel_original/README.md).

### Long jobs and resuming

Every package saves `model_last.mdl` after each epoch. Pass `--resume` (optionally `--resume-path`) to continue from the epoch after the last saved one. This works for time-limited environments (for example notebook platforms), where training is stopped before the time limit and restarted from the saved checkpoint.

## Comparing methods fairly

The packages follow the original implementations where those differ, so check the following before putting numbers side by side:

| | `Baseline` (RAA-KGC) | `SimKGC_Restructured` | `HaSa_Restructured` | `ARPM_KGC_DataParallel_original` |
| --- | --- | --- | --- | --- |
| Link graph (neighbour text, hop anchors) | opt-in `--use-link-graph` | opt-in `--use-link-graph` | opt-in `--use-link-graph` | **on** by default (`--disable-link-graph`) |
| Default re-rank (`--neighbor-weight`, `--rerank-n-hop`) | 0.02, 2 | 0.05, 2 (`eval.sh`: 5 hops on WN18RR) | 0.05, 2 (`eval_hasa.sh`: 0 for HaSa) | 0.05, 5 |
| Extra filtering of entities that have the head as a tail | yes | no | no | yes |
| Default pooling | mean | cls (scripts use mean) | cls (HaSa scripts: pooler) | mean |

Use the same `--neighbor-weight` / `--rerank-n-hop` and the same train/validation/test files for every method, and report the metric of the same checkpoint-selection rule. Reported numbers are the average of the forward and backward filtered metrics on the test split.

## Results

Fill in after your final runs (test split, filtered, average of both directions).

| Method | WN18RR MRR | Hits@1 | Hits@10 | FB15k-237 MRR | Hits@1 | Hits@10 |
| --- | --- | --- | --- | --- | --- | --- |
| SimKGC | | | | | | |
| HaSa | | | | | | |
| RAA-KGC | | | | | | |
| **ARPM-KGC** | | | | | | |

## Citation and acknowledgements

If you use this code, please cite our paper (reference to be added upon publication) and the works it builds on:

- Wang et al., *SimKGC: Simple Contrastive Knowledge Graph Completion with Pre-trained Language Models*, ACL 2022.
- Yuan et al., *Knowledge Graph Completion with Relation-Aware Anchor Enhancement*, AAAI 2025.
- The HaSa method (hardness- and structure-aware contrastive knowledge graph embedding).

## License

No license is currently specified. Add one before public release or redistribution.

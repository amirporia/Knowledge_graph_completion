# ARPM-KGC on top of the RAA-KGC Baseline

The code base is the **Baseline** (RAA-KGC) project. The ARPM architecture from
`ARPM_KGC_DataParallel_original` is added as a *switchable extension*, so the baseline and the
extended model run through the **same** data pipeline, trainer, predictor and evaluator. The only
difference between the two runs is `--disable-memory`.

```
python -m ARPM_KGC.preprocess.preprocess --task wn18rr      # data in <root>/data/<task>/

# Baseline (exactly RAA-KGC: no memory module is even instantiated)
python -m ARPM_KGC.main --task wn18rr --disable-memory
python -m ARPM_KGC.evaluation.evaluate --task wn18rr --is-test --disable-memory

# Baseline + ARPM
python -m ARPM_KGC.main --task wn18rr
python -m ARPM_KGC.evaluation.evaluate --task wn18rr --is-test
# multi-GPU: torchrun --nproc_per_node=2 -m ARPM_KGC.main --task wn18rr
```

Checkpoints go to `data/<task>/checkpoint_baseline/` and `data/<task>/checkpoint_arpm/`.
Pass the same `--disable-memory` flag when evaluating the baseline (or give `--eval-model-path`).

**Fair comparison:** use `--checkpoint-metric mrr` for both runs. The default `acc` is the
baseline's in-batch Acc@1, which only measures the encoder query and cannot see the memory branch.

## What was ported from ARPM

| Piece | File | Notes |
| --- | --- | --- |
| Candidate pool A(h,r) (local hops 0..N + global) | `utils/candidate_pool.py` | train graph only, no label leakage |
| `LinkGraph.get_hop_layers` | `utils/triplet.py` | layer-wise BFS, hides the query's own edge in training |
| Candidate tail tokens, memory fields in `collate` | `utils/doc.py` | only built when memory is on |
| ProtoGen, HopScorer, MemoryGate, diversity losses | `model/modules.py` | unchanged |
| Anchor attention α, prototypes, structural memory, gate, `S_p`/`S_struct` | `model/models.py` | `_memory*`, `score_*` |
| L_proto, L_struct, L_combined, L_div, L_pdiv + separate memory LR | `model/trainer.py` | encoders get exactly the baseline losses |
| Score `S = S_q + S_hrta + λp S_p + λs S_struct`, scale tuning | `evaluation/evaluate.py` | `--tune-scales` includes (0,0) = baseline |

## What stays baseline

* RAA anchors (`doc.sample_anchors`, `anchor_num`, Eq. 5 mean of anchor queries), `L_hr`, `L_hrta`
  (with its own `related_triplet_mask`), self-negative, temperature, baseline defaults for
  `--rerank-n-hop 2` / `--neighbor-weight 0.02`, `--use-link-graph` text augmentation (off by default).
* In ARPM-original the RAA anchors were taken from the memory pool; here they are still drawn by the
  baseline sampler, and the memory pool is an *independent* extra input. This isolates the memory
  branch as the only change (cost: candidates are encoded in addition to the RAA anchors).

## Deliberate differences from the original Baseline (apply to BOTH runs)

* Evaluation filter uses `-1e4` instead of `-1` and an O(1) reverse index; ranking = `1 + #(score > gold)`.
  Same filter set, but absolute numbers can differ slightly from the original baseline script.
* Optional full filtered-ranking checkpoint selection (`--checkpoint-metric mrr|hit@k`).

## Ablations (flags)

`--random-anchor-selection` (A1), `--eta-div 0` (A2), `--num-prototypes 1` (A3), `--uniform-hop-weighting` (A5),
`--global-budget 0` (A6), `--disable-hop-anchors` (A7), `--fixed-lambda-p/-s 0.5` (A8), `--fixed-lambda-p 0` (A9),
`--fixed-lambda-s 0` (A10), `--eta-combined 0` (frozen gate diagnostic).

## Status

Python files compile. **Not executed**: this sandbox cannot install torch/transformers, so no smoke test
was run. Run a 1-epoch `--disable-memory` and a default run on WN18RR first and check the logs
(`lam_p`, `lam_s`, `L_proto`, `L_struct`, `L_comb` appear only in the ARPM run).

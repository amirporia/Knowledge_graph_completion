"""
Config presets for the ablation study of Sec 7.4 / Table 3, plus the
faithful-baseline reconstruction used as Table 3's reference row.

Every preset starts from `full_config()` -- the core ARPM-KGC model with
every core mechanism (Sec 5, Table 2's "core" column) switched on and every
optional Sec 4.13 discrete extension switched off -- and flips exactly the
flag(s) the corresponding ablation targets.

Design decisions worth flagging explicitly (the proposal text itself is a
plan, not a fully pinned-down spec, so these mappings are our documented
reading of it):

  * A6 "No global relation memory" vs A7 "No local structural memory":
    both are read as *candidate-source* ablations (Sec 4.3): A6 removes
    A_global(r) from the retrieval pool, A7 removes A_local(h, r). A7's
    removal of local candidates is also the more literal way to realize
    "no local structural memory", since Sec 4.6 defines structural memory
    as built exclusively from local anchors -- taking away the local
    candidate source necessarily empties it.
  * A9 "No semantic prototype memory" vs A10 "No structural memory": read
    as the two *fusion-branch* ablations at G_mem (Sec 4.8) -- these are
    the pair Sec 7.4 calls out explicitly ("test whether the two memory
    sources are complementary").
  * A4 "Fixed prototype count instead of adaptive K" is the natural control
    for A13 (adaptive K via Sec 4.13.3): since the core model already uses
    a fixed K (Sec 4.5.1 -- adaptive K is entirely optional), A4 is
    equivalent to `full_config()`, kept as a distinct named preset so it
    reads clearly alongside A13 in results tables.
  * `hop_score_uses_prototype_query` is left at its config default (False,
    matching Algorithm 1) in every preset below; see config.py's docstring
    for the qp-vs-q ambiguity this resolves.
"""

import copy
from typing import Callable, Dict

from .config import ARPMConfig


def full_config(**overrides) -> ARPMConfig:
    """Full core ARPM-KGC (Table 3's 'Full' row): every core mechanism on,
    every Sec 4.13 discrete extension off."""
    cfg = ARPMConfig(ablation_name="full")
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def baseline_faithful_config(**overrides) -> ARPMConfig:
    """Best-effort faithful reconstruction of the cited RAA-KGC baseline
    (Table 3's 'Baseline' row), built from the proposal's own
    characterization of it (Sec 1: "generates anchor entities from a
    relation-aware neighborhood and uses them to make the query
    representation more discriminative"). Concretely: a single
    unconditional anchor-average memory vector (no learned relevance
    scoring, no multi-prototype decomposition, no structural/hop
    adaptivity, no adaptive gate) is always added to the query. Since the
    cited paper's own implementation details are not part of this
    proposal, this preset is deliberately the *simplest* instantiation of
    "anchor-enhanced query" consistent with that description -- it is not
    a line-for-line reproduction of any specific external codebase.
    """
    cfg = ARPMConfig(ablation_name="baseline_faithful")
    cfg.anchor_selection_mode = "uniform"          # no query-conditioned scoring
    cfg.anchor_gating_mode = "none"
    cfg.use_diversity_regularization = False
    cfg.use_retrieval_supervision = False
    cfg.use_prototype_memory = True
    cfg.num_prototypes = 1                          # single memory vector, no prototype decomposition
    cfg.weight_prototype_attention_by_alpha = False
    cfg.prototype_gating_mode = "fixed"
    cfg.use_prototype_loss = False
    cfg.use_structural_memory = False               # no hop-adaptive structural memory
    cfg.use_adaptive_memory_gate = False             # unconditional memory injection
    cfg.fixed_memory_gate = 1.0
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def a1_random_anchors(**overrides) -> ARPMConfig:
    """A1: Random anchors instead of query-conditioned selection."""
    cfg = full_config(ablation_name="A1_random_anchors")
    cfg.anchor_selection_mode = "random"
    return _apply(cfg, overrides)


def a2_no_diversity_regularization(**overrides) -> ARPMConfig:
    """A2: No diversity regularization (eta_div's contribution removed)."""
    cfg = full_config(ablation_name="A2_no_diversity_reg")
    cfg.use_diversity_regularization = False
    return _apply(cfg, overrides)


def a3_single_prototype(**overrides) -> ARPMConfig:
    """A3: Single prototype, K = 1."""
    cfg = full_config(ablation_name="A3_single_prototype")
    cfg.num_prototypes = 1
    return _apply(cfg, overrides)


def a4_fixed_prototype_count(**overrides) -> ARPMConfig:
    """A4: Fixed prototype count instead of adaptive K (control for A13;
    equivalent to `full_config()` -- see module docstring)."""
    cfg = full_config(ablation_name="A4_fixed_prototype_count")
    cfg.prototype_gating_mode = "fixed"
    return _apply(cfg, overrides)


def a5_uniform_hop_weighting(**overrides) -> ARPMConfig:
    """A5: Fixed/uniform hop weighting instead of adaptive beta_l."""
    cfg = full_config(ablation_name="A5_uniform_hop_weighting")
    cfg.hop_selection_mode = "uniform"
    return _apply(cfg, overrides)


def a6_no_global_relation_memory(**overrides) -> ARPMConfig:
    """A6: No global relation memory (A_global(r) removed from the pool)."""
    cfg = full_config(ablation_name="A6_no_global_relation_memory")
    cfg.use_global_candidates = False
    return _apply(cfg, overrides)


def a7_no_local_structural_memory(**overrides) -> ARPMConfig:
    """A7: No local structural memory (A_local(h, r) removed from the pool,
    which also empties structural memory since it is local-only by
    construction, Sec 4.6)."""
    cfg = full_config(ablation_name="A7_no_local_structural_memory")
    cfg.use_local_candidates = False
    return _apply(cfg, overrides)


def a8_fixed_memory_gate(**overrides) -> ARPMConfig:
    """A8: Fixed memory contribution instead of adaptive gate lambda_mem."""
    cfg = full_config(ablation_name="A8_fixed_memory_gate")
    cfg.use_adaptive_memory_gate = False
    cfg.fixed_memory_gate = overrides.pop("fixed_memory_gate", 0.5)
    return _apply(cfg, overrides)


def a9_no_semantic_prototype_memory(**overrides) -> ARPMConfig:
    """A9: No semantic prototype memory (G_mem fuses structural memory only)."""
    cfg = full_config(ablation_name="A9_no_prototype_memory")
    cfg.use_prototype_memory = False
    cfg.use_prototype_loss = False
    return _apply(cfg, overrides)


def a10_no_structural_memory(**overrides) -> ARPMConfig:
    """A10: No structural memory (G_mem fuses prototype memory only)."""
    cfg = full_config(ablation_name="A10_no_structural_memory")
    cfg.use_structural_memory = False
    return _apply(cfg, overrides)


def a11_discrete_anchor_gating(**overrides) -> ARPMConfig:
    """A11: Gumbel-Sigmoid discrete anchor retrieval gate (Sec 4.13.1)."""
    cfg = full_config(ablation_name="A11_discrete_anchor_gating")
    cfg.anchor_gating_mode = "gumbel_sigmoid"
    return _apply(cfg, overrides)


def a12_discrete_hop_selection(**overrides) -> ARPMConfig:
    """A12: Gumbel-Softmax discrete hop selection (Sec 4.13.2)."""
    cfg = full_config(ablation_name="A12_discrete_hop_selection")
    cfg.hop_selection_mode = "gumbel_softmax"
    cfg.top_k_hop = overrides.pop("top_k_hop", 1)
    return _apply(cfg, overrides)


def a13_adaptive_prototype_gating(**overrides) -> ARPMConfig:
    """A13: Gumbel-Sigmoid adaptive prototype slot gating (Sec 4.13.3)."""
    cfg = full_config(ablation_name="A13_adaptive_prototype_gating")
    cfg.prototype_gating_mode = "gumbel_sigmoid_slots"
    return _apply(cfg, overrides)


def _apply(cfg: ARPMConfig, overrides: dict) -> ARPMConfig:
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


ABLATIONS: Dict[str, Callable[..., ARPMConfig]] = {
    "baseline": baseline_faithful_config,
    "full": full_config,
    "A1": a1_random_anchors,
    "A2": a2_no_diversity_regularization,
    "A3": a3_single_prototype,
    "A4": a4_fixed_prototype_count,
    "A5": a5_uniform_hop_weighting,
    "A6": a6_no_global_relation_memory,
    "A7": a7_no_local_structural_memory,
    "A8": a8_fixed_memory_gate,
    "A9": a9_no_semantic_prototype_memory,
    "A10": a10_no_structural_memory,
    "A11": a11_discrete_anchor_gating,
    "A12": a12_discrete_hop_selection,
    "A13": a13_adaptive_prototype_gating,
}


def get_ablation_config(name: str, **overrides) -> ARPMConfig:
    if name not in ABLATIONS:
        raise ValueError(f"Unknown ablation '{name}'. Choices: {sorted(ABLATIONS)}")
    return ABLATIONS[name](**overrides)


def all_ablation_configs() -> Dict[str, ARPMConfig]:
    return {name: factory() for name, factory in ABLATIONS.items()}

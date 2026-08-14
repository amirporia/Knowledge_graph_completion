"""
Central configuration for ARPM-KGC.

Every switch in the proposal's "core vs optional" distinction (Section 5,
Table 2) and every row of the planned ablation study (Section 7.4, Table 3)
is represented here as an explicit field, so a full experiment is fully
described by an `ARPMConfig` instance -- no hidden flags, no code edits.

Field groups mirror the proposal's section numbering (noted in comments)
so the mapping from paper -> config -> code stays traceable.

Ready-made presets that reproduce Table 3's rows are provided in
`arpmkgc/ablations.py`; this module only defines the schema and CLI parsing.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

PACKAGE_ROOT = Path(__file__).parent.parent.absolute()

# Anchor-selection strategies (Section 4.4; A1 in Table 3)
ANCHOR_SELECTION_MODES = ("learned", "random", "uniform")
# Anchor keep/drop gating on top of soft weights (Section 4.13.1; A11)
ANCHOR_GATING_MODES = ("none", "gumbel_sigmoid")
# Hop-weighting strategies for structural memory (Section 4.6; A5, 4.13.2/A12)
HOP_SELECTION_MODES = ("soft", "uniform", "gumbel_softmax")
# Prototype-count strategies (Section 4.5.1; A13/4.13.3)
PROTOTYPE_GATING_MODES = ("fixed", "gumbel_sigmoid_slots")
SUPPORTED_TASKS = ("wn18rr", "fb15k237")


@dataclass
class ARPMConfig:
    # ------------------------------------------------------------------
    # Task / data
    # ------------------------------------------------------------------
    task: str = "wn18rr"
    train_path: Optional[str] = None
    valid_path: Optional[str] = None
    test_path: Optional[str] = None
    model_dir: Optional[str] = None
    eval_model_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Encoders E0 (query / anchor-enhanced encoder) and E1 (entity encoder)
    # (Section 4.2)
    # ------------------------------------------------------------------
    pretrained_model: str = "bert-base-uncased"
    pooling: str = "mean"          # 'cls' | 'mean' | 'max'
    dropout: float = 0.1
    max_num_tokens: int = 50
    tie_encoders: bool = False     # if True, E0 and E1 share weights at init only (still two towers)

    # ------------------------------------------------------------------
    # 4.3 Relation-aware candidate memory (local + global candidate pool)
    # ------------------------------------------------------------------
    max_hop: int = 2                       # L: max structural hop distance considered
    candidate_budget: int = 32             # M: total anchors kept before neural scoring
    local_candidates_per_hop: int = 8      # cap on local candidates sampled per hop
    global_candidates: int = 16            # cap on global relation-level candidates
    use_local_candidates: bool = True      # A7: "No local structural memory" -> False
    use_global_candidates: bool = True     # A6: "No global relation memory" -> False

    # ------------------------------------------------------------------
    # 4.4 Query-conditioned anchor selection
    # ------------------------------------------------------------------
    anchor_selection_mode: str = "learned"      # A1: "learned" -> "random"
    retrieval_temperature: float = 1.0          # tau_r
    use_diversity_regularization: bool = True   # A2 -> False (L_div term dropped)
    diversity_weight: float = 0.1               # eta_div
    anchor_gating_mode: str = "none"            # A11: "none" -> "gumbel_sigmoid"

    # optional explicit retrieval supervision (L_rel); off by default because
    # the proposal notes retrieval can be learned indirectly through the
    # downstream KGC objective when explicit anchor labels are unavailable
    # (Section 4.4.1). L_retr = L_rel + eta_div * L_div (Sec 4.4.1); note
    # eta_r below scales the *whole* L_retr sum (Sec 4.11), so it still
    # applies to the diversity term even when L_rel itself is disabled.
    use_retrieval_supervision: bool = False
    retrieval_loss_weight: float = 1.0          # eta_r, scales L_retr = L_rel + eta_div * L_div

    # ------------------------------------------------------------------
    # 4.5 Multi-prototype semantic memory
    # ------------------------------------------------------------------
    use_prototype_memory: bool = True           # A9: "No semantic prototype memory" -> False
    num_prototypes: int = 4                     # K in {1, 2, 4, 8}; A3 sets this to 1
    weight_prototype_attention_by_alpha: bool = True  # incorporate alpha_i into rho_ik (4.5)
    prototype_gating_mode: str = "fixed"        # A13: "fixed" -> "gumbel_sigmoid_slots"
    max_prototype_slots: int = 8                # K_max for the 4.13.3 extension

    # prototype-consistency auxiliary loss L_p (Section 4.10 / 4.11)
    use_prototype_loss: bool = True
    prototype_loss_weight: float = 0.2          # eta_p
    prototype_temperature: float = 0.1          # tau_p (log-sum-exp pooling temperature)

    # ------------------------------------------------------------------
    # 4.6 Adaptive structural memory
    # ------------------------------------------------------------------
    use_structural_memory: bool = True          # A10: "No structural memory" -> False
    hop_selection_mode: str = "soft"            # A5: "soft" -> "uniform"; A12 -> "gumbel_softmax"
    top_k_hop: int = 1                          # k_hop for the Gumbel-top-k hop-selection variant
    # Resolves the qp-vs-q ambiguity between Section 4.6's prose (uses qp,
    # "the query representation after prototype interaction") and Algorithm 1
    # (line 15 scores hops with the raw query q, computed *before* qp exists
    # at line 19). We default to Algorithm 1 -- the executable procedure --
    # and expose the prose reading as an opt-in flag.
    hop_score_uses_prototype_query: bool = False
    structural_eps: float = 1e-8                # epsilon in 4.6's hop aggregation

    # ------------------------------------------------------------------
    # 4.10 Adaptive memory-aware tail prediction
    # ------------------------------------------------------------------
    use_adaptive_memory_gate: bool = True       # A8: "Fixed memory contribution" -> False
    fixed_memory_gate: float = 0.5              # lambda_mem used when the gate is not adaptive
    score_temperature: float = 0.05             # inverse-temperature init for sim(.,.) (as in tau of 4.2)
    finetune_score_temperature: bool = True

    # ------------------------------------------------------------------
    # 4.11 Training objective / in-batch negatives
    # ------------------------------------------------------------------
    additive_margin: float = 0.02
    use_self_negative: bool = True

    # ------------------------------------------------------------------
    # 4.13 Gumbel-Softmax / Gumbel-Sigmoid discrete extensions
    # (shared temperature-annealing schedule; can also be tuned per-mechanism)
    # ------------------------------------------------------------------
    gumbel_temperature_init: float = 1.0
    gumbel_temperature_min: float = 0.1
    gumbel_temperature_anneal_rate: float = 0.97   # multiplicative decay per epoch
    gumbel_sel_temperature: Optional[float] = None    # tau_sel override; None -> shared schedule
    gumbel_hop_temperature: Optional[float] = None    # tau_hop override
    gumbel_proto_temperature: Optional[float] = None  # tau_proto override

    # ------------------------------------------------------------------
    # Training / optimization
    # ------------------------------------------------------------------
    epochs: int = 20
    batch_size: int = 16
    lr: float = 5e-5
    weight_decay: float = 1e-4
    lr_scheduler: str = "linear"    # 'linear' | 'cosine'
    warmup: int = 400
    grad_clip: float = 10.0
    seed: Optional[int] = None
    max_to_keep: int = 4
    eval_every_n_step: int = 10000
    resume: bool = False
    resume_path: Optional[str] = None

    # ------------------------------------------------------------------
    # System
    # ------------------------------------------------------------------
    workers: int = 4
    print_freq: int = 20
    use_amp: bool = False
    gpu: int = 0
    dist_backend: str = "nccl"
    is_test: bool = False

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------
    ablation_name: str = "full"     # which Table-3 row this config instantiates
    run_name: Optional[str] = None
    eval_split: str = "valid"       # which split `arpmkgc.evaluate` scores against

    # Populated by setup_distributed(); not user-facing CLI flags.
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    distributed: bool = False

    def validate(self) -> None:
        assert self.task in SUPPORTED_TASKS, f"Unknown task: {self.task}"
        assert self.anchor_selection_mode in ANCHOR_SELECTION_MODES
        assert self.anchor_gating_mode in ANCHOR_GATING_MODES
        assert self.hop_selection_mode in HOP_SELECTION_MODES
        assert self.prototype_gating_mode in PROTOTYPE_GATING_MODES
        assert self.num_prototypes >= 1
        assert self.max_prototype_slots >= self.num_prototypes
        assert self.max_hop >= 1
        assert 0.0 <= self.fixed_memory_gate <= 1.0
        assert self.top_k_hop >= 1
        if self.hop_selection_mode == "gumbel_softmax":
            assert self.top_k_hop <= self.max_hop
        if not self.use_local_candidates and self.use_structural_memory:
            warnings.warn(
                "use_structural_memory=True but use_local_candidates=False: "
                "structural memory has no local anchors to draw from and "
                "will always be an epsilon-stabilized zero vector."
            )
        if not self.use_prototype_memory and not self.use_structural_memory:
            warnings.warn(
                "Both use_prototype_memory and use_structural_memory are "
                "False: the memory branch is fully disabled and the model "
                "degenerates to the plain query/entity dual encoder."
            )

    def resolve_paths(self) -> "ARPMConfig":
        """Fill in default data/checkpoint paths from `task`, mirroring the
        directory layout produced by `arpmkgc.preprocess`."""
        task_dir = PACKAGE_ROOT / "data" / self.task.upper()
        if self.train_path is None:
            self.train_path = str(task_dir / "train.txt.json")
        if self.valid_path is None:
            self.valid_path = str(task_dir / "valid.txt.json")
        if self.test_path is None:
            self.test_path = str(task_dir / "test.txt.json")
        if self.model_dir is None:
            run = self.run_name or self.ablation_name
            self.model_dir = str(task_dir / "checkpoint" / run)
        if self.eval_model_path is None:
            self.eval_model_path = str(Path(self.model_dir) / "model_best.mdl")
        return self

    def setup_environment(self) -> "ARPMConfig":
        if self.seed is not None:
            random.seed(self.seed)
            try:
                import torch
                torch.manual_seed(self.seed)
                import torch.backends.cudnn as cudnn
                cudnn.deterministic = True
            except ImportError:
                pass
        try:
            import torch
            if not torch.cuda.is_available():
                self.use_amp = False
                self.print_freq = max(self.print_freq, 1)
        except ImportError:
            self.use_amp = False
        return self

    def setup_distributed(self) -> "ARPMConfig":
        """Detect torchrun's environment variables (RANK/LOCAL_RANK/WORLD_SIZE).

        A plain `python -m arpmkgc.main` leaves them unset, which is treated
        as a single-process run (world_size=1, rank=0).
        """
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.distributed = self.world_size > 1
        return self

    def finalize(self, make_dirs: bool = True) -> "ARPMConfig":
        self.resolve_paths()
        self.validate()
        self.setup_environment()
        self.setup_distributed()
        if make_dirs:
            os.makedirs(self.model_dir, exist_ok=True)
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "ARPMConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # drop runtime-only fields so old checkpoints stay loadable
        for runtime_field in ("world_size", "rank", "local_rank", "distributed"):
            data.pop(runtime_field, None)
        return cls(**data)


def _add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    dest = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    group.add_argument(f"--no-{name}", dest=dest, action="store_false", help=argparse.SUPPRESS)
    parser.set_defaults(**{dest: default})


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = ARPMConfig()
    parser = argparse.ArgumentParser(description="ARPM-KGC")

    parser.add_argument("--ablation", type=str, default=None,
                         help="Name of a preset from arpmkgc.ablations (overrides individual flags "
                              "with the preset, then any explicitly-passed flags win).")

    data = parser.add_argument_group("Data")
    data.add_argument("--task", default=defaults.task, choices=SUPPORTED_TASKS)
    data.add_argument("--train-path", default=None)
    data.add_argument("--valid-path", default=None)
    data.add_argument("--test-path", default=None)
    data.add_argument("--model-dir", default=None)
    data.add_argument("--eval-model-path", default=None)
    data.add_argument("--run-name", default=None)
    data.add_argument("--eval-split", default="valid", choices=["valid", "test"],
                       help="Which split `arpmkgc.evaluate` scores against.")

    enc = parser.add_argument_group("Encoders (Sec 4.2)")
    enc.add_argument("--pretrained-model", default=defaults.pretrained_model)
    enc.add_argument("--pooling", default=defaults.pooling, choices=["cls", "mean", "max"])
    enc.add_argument("--dropout", type=float, default=defaults.dropout)
    enc.add_argument("--max-num-tokens", type=int, default=defaults.max_num_tokens)

    cand = parser.add_argument_group("Candidate memory (Sec 4.3)")
    cand.add_argument("--max-hop", type=int, default=defaults.max_hop)
    cand.add_argument("--candidate-budget", type=int, default=defaults.candidate_budget)
    cand.add_argument("--local-candidates-per-hop", type=int, default=defaults.local_candidates_per_hop)
    cand.add_argument("--global-candidates", type=int, default=defaults.global_candidates)
    _add_bool_arg(cand, "use-local-candidates", defaults.use_local_candidates, "A7 control")
    _add_bool_arg(cand, "use-global-candidates", defaults.use_global_candidates, "A6 control")

    sel = parser.add_argument_group("Anchor selection (Sec 4.4, 4.13.1)")
    sel.add_argument("--anchor-selection-mode", default=defaults.anchor_selection_mode,
                      choices=ANCHOR_SELECTION_MODES, help="A1 control")
    sel.add_argument("--retrieval-temperature", type=float, default=defaults.retrieval_temperature)
    _add_bool_arg(sel, "use-diversity-regularization", defaults.use_diversity_regularization, "A2 control")
    sel.add_argument("--diversity-weight", type=float, default=defaults.diversity_weight)
    sel.add_argument("--anchor-gating-mode", default=defaults.anchor_gating_mode,
                      choices=ANCHOR_GATING_MODES, help="A11 control")
    _add_bool_arg(sel, "use-retrieval-supervision", defaults.use_retrieval_supervision, "")
    sel.add_argument("--retrieval-loss-weight", type=float, default=defaults.retrieval_loss_weight)

    proto = parser.add_argument_group("Prototype memory (Sec 4.5, 4.13.3)")
    _add_bool_arg(proto, "use-prototype-memory", defaults.use_prototype_memory, "A9 control")
    proto.add_argument("--num-prototypes", type=int, default=defaults.num_prototypes, help="A3, A4 control")
    _add_bool_arg(proto, "weight-prototype-attention-by-alpha",
                  defaults.weight_prototype_attention_by_alpha, "")
    proto.add_argument("--prototype-gating-mode", default=defaults.prototype_gating_mode,
                        choices=PROTOTYPE_GATING_MODES, help="A13 control")
    proto.add_argument("--max-prototype-slots", type=int, default=defaults.max_prototype_slots)
    _add_bool_arg(proto, "use-prototype-loss", defaults.use_prototype_loss, "")
    proto.add_argument("--prototype-loss-weight", type=float, default=defaults.prototype_loss_weight)
    proto.add_argument("--prototype-temperature", type=float, default=defaults.prototype_temperature)

    struct = parser.add_argument_group("Structural memory (Sec 4.6, 4.13.2)")
    _add_bool_arg(struct, "use-structural-memory", defaults.use_structural_memory, "A10 control")
    struct.add_argument("--hop-selection-mode", default=defaults.hop_selection_mode,
                         choices=HOP_SELECTION_MODES, help="A5, A12 control")
    struct.add_argument("--top-k-hop", type=int, default=defaults.top_k_hop)
    _add_bool_arg(struct, "hop-score-uses-prototype-query",
                  defaults.hop_score_uses_prototype_query, "")

    pred = parser.add_argument_group("Prediction (Sec 4.10)")
    _add_bool_arg(pred, "use-adaptive-memory-gate", defaults.use_adaptive_memory_gate, "A8 control")
    pred.add_argument("--fixed-memory-gate", type=float, default=defaults.fixed_memory_gate)
    pred.add_argument("--score-temperature", type=float, default=defaults.score_temperature)

    train = parser.add_argument_group("Training")
    train.add_argument("--epochs", type=int, default=defaults.epochs)
    train.add_argument("-b", "--batch-size", type=int, default=defaults.batch_size)
    train.add_argument("--lr", type=float, default=defaults.lr)
    train.add_argument("--wd", "--weight-decay", type=float, default=defaults.weight_decay, dest="weight_decay")
    train.add_argument("--lr-scheduler", default=defaults.lr_scheduler, choices=["linear", "cosine"])
    train.add_argument("--warmup", type=int, default=defaults.warmup)
    train.add_argument("--grad-clip", type=float, default=defaults.grad_clip)
    train.add_argument("--seed", type=int, default=None)
    train.add_argument("--max-to-keep", type=int, default=defaults.max_to_keep)
    train.add_argument("--eval-every-n-step", type=int, default=defaults.eval_every_n_step)
    train.add_argument("--resume", action="store_true", default=False)
    train.add_argument("--resume-path", default=None)

    sysg = parser.add_argument_group("System")
    sysg.add_argument("-j", "--workers", type=int, default=defaults.workers)
    sysg.add_argument("-p", "--print-freq", type=int, default=defaults.print_freq)
    _add_bool_arg(sysg, "use-amp", defaults.use_amp, "")
    sysg.add_argument("--gpu", type=int, default=defaults.gpu)

    return parser


def parse_args(argv=None) -> ARPMConfig:
    from .ablations import get_ablation_config, ABLATIONS

    parser = build_arg_parser()
    ns = parser.parse_args(argv)
    ns_dict = vars(ns)
    ablation_name = ns_dict.pop("ablation")

    if ablation_name is not None:
        assert ablation_name in ABLATIONS, (
            f"Unknown ablation '{ablation_name}'. Choices: {sorted(ABLATIONS)}"
        )
        cfg = get_ablation_config(ablation_name)
        # Only overwrite with CLI flags the user actually supplied (argparse
        # gives no clean way to detect that generically here, so we accept
        # that explicit --flags always win over the preset by re-applying
        # every parsed value; presets are meant to be used largely as-is,
        # with paths/system flags customized per run).
        cli_overrides = {
            k: v for k, v in ns_dict.items()
            if k in ("train_path", "valid_path", "test_path", "model_dir",
                      "eval_model_path", "run_name", "epochs", "batch_size",
                      "lr", "seed", "workers", "use_amp", "gpu",
                      "resume", "resume_path", "eval_split", "print_freq",
                      "max_to_keep", "eval_every_n_step")
            and v is not None
        }
        for k, v in cli_overrides.items():
            setattr(cfg, k, v)
    else:
        cfg = ARPMConfig(**ns_dict)

    return cfg.finalize()

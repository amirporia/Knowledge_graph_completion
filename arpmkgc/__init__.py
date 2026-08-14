"""
ARPM-KGC: Adaptive Relation-Aware Prototype Memory for Knowledge Graph Completion.

Independent, from-scratch implementation of the ARPM-KGC architecture and
experimental plan described in the accompanying research proposal
("Adaptive Relation-Aware Prototype Memory for Knowledge Graph Completion").

This package is intentionally self-contained: it does not import anything
from the separate baseline codebase (`ours/`). Where the proposal is silent
about an implementation detail (data preprocessing conventions, checkpoint
format, distributed-training scaffolding, ...) this package makes its own
independent design choice, documented in README.md and in the relevant
module docstring.

Every optional mechanism described in the proposal (Sections 4.4-4.13, and
Table 3's ablations A1-A13) is exposed as a boolean/enum field on
`ARPMConfig` (see `arpmkgc/config.py`) so experiments can be toggled without
code changes. `arpmkgc/ablations.py` provides ready-made presets for the
full ablation study in Table 3.
"""

__version__ = "0.1.0"

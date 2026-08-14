"""Standalone logger configuration for ARPM-KGC (no dependency on `ours`)."""

import logging


def _build_logger() -> logging.Logger:
    fmt = logging.Formatter("[%(asctime)s %(levelname)s %(name)s] %(message)s")
    log = logging.getLogger("arpmkgc")
    log.setLevel(logging.INFO)
    log.propagate = False

    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        log.addHandler(handler)

    return log


logger = _build_logger()

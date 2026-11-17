"""Stage-2 Ray Syncer simulation: pre-batching baseline vs PR #57641 batching.

This package is a *copy* of the stage-1 ``raysyncer_sim`` package with the
send path of PR #57641 added.  The stage-1 package is left untouched; a diff
of ``core.py`` shows exactly the batching delta.
"""

from .config import SimConfig
from .core import Simulator, SimResult

__all__ = ["SimConfig", "Simulator", "SimResult"]


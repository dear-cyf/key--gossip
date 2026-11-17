"""Pure-Python discrete-event reproduction of the Ray Syncer propagation
problem described in ray-project/ray PR #57641 (pre-batching baseline).

See ``README.md`` in the repository root for the model description.
"""

from .config import SimConfig
from .core import Simulator, SimResult

__all__ = ["SimConfig", "Simulator", "SimResult"]


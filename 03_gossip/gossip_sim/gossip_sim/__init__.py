"""Gossip propagation experiment for RaySyncer.

This package reuses, **unmodified**, the resource-update generator, the
network/executor cost model and the load model of the existing Ray Syncer
simulations:

* ``stage1_simulation/raysyncer_sim`` -- pre-batching star (the published
  baseline of PR #57641),
* ``stage2_batching_sim/batch_sim``   -- star + sync-message batching.

Only the propagation protocol is new: ``gossip_sim.core`` replaces the
single-GCS fan-out with a k-neighbour overlay where every Raylet relays to
``f`` of its neighbours.
"""

from .params import (
    GossipParams,
    estimate_message_bytes,
    stage1_config_class,
    stage2_config_class,
)
from .topology import GossipTopology
from .recorder import MetricRecorder

__all__ = [
    "GossipParams",
    "GossipTopology",
    "MetricRecorder",
    "estimate_message_bytes",
    "stage1_config_class",
    "stage2_config_class",
]

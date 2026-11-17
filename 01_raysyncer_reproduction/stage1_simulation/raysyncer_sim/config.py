"""Tunable parameters of the Ray Syncer simulation.

Timing parameters are expressed in **microseconds**; simulation-window knobs
are in seconds.  The defaults are calibrated against the 1,000-node
measurement published in ray-project/ray PR #57641 (first wave ~150 s, full
convergence with the second-wave feedback ~10 minutes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

US = 1e-6  # microseconds -> seconds


@dataclass
class SimConfig:
    """Configuration of a single Ray-Syncer propagation experiment."""

    # ------------------------------------------------------------------ #
    # Cluster shape
    # ------------------------------------------------------------------ #
    num_nodes: int = 300
    seed: int = 20251011

    # ------------------------------------------------------------------ #
    # Wave 1 -- "every Raylet creates a placement group and immediately
    # reports its changed resource view to the GCS".
    # ------------------------------------------------------------------ #
    wave1_start_s: float = 0.5
    wave1_emit_spread_s: float = 0.05

    # ------------------------------------------------------------------ #
    # Costs on the single-threaded GCS event loop.
    # ------------------------------------------------------------------ #
    # Per inbound message: NodeState::ConsumeSyncMessage + io_context dispatch.
    gcs_fixed_us: float = 10.0
    # Per connected reactor inside RaySyncer::BroadcastMessage's fan-out loop
    # (PushToSendingQueue).  The loop runs N-1 times per message -- this is
    # where the O(N^2) amplification comes from.
    gcs_fanout_us: float = 1.0
    # Per completed gRPC write on a connection.  RaySyncerBidiReactorBase
    # writes exactly one message at a time (the ``sending_`` flag), so each
    # completed write also costs one GCS event-loop slot.
    gcs_write_us: float = 150.0
    # Per-connection spread of the write cost (gRPC streams are not identical):
    # each connection is scaled by 1 +/- link_write_jitter/2, fixed for the run.
    link_write_jitter: float = 0.25
    # Keep the real (node_id, message_type) de-duplication of sending_buffer_.
    # Set to False to model the raw FIFO "message channel" the PR describes.
    gcs_reactor_dedup: bool = True

    # ------------------------------------------------------------------ #
    # Network / Raylet costs
    # ------------------------------------------------------------------ #
    uplink_us: float = 50.0      # Raylet -> GCS one-way hop
    prop_us: float = 500.0       # GCS -> Raylet one-way hop
    node_proc_us: float = 100.0  # Raylet applying an update to its local view

    # ------------------------------------------------------------------ #
    # Deprecated historical wave-2 knobs, retained for old config loading and
    # jitter RNG compatibility. Coverage thresholds no longer emit messages.
    # Use resource_changes for any second or subsequent local changes.
    # ------------------------------------------------------------------ #
    wave2_enabled: bool = False
    wave2_milestones: Tuple[float, ...] = (0.02, 0.05, 0.10, 0.20, 0.35, 0.55, 0.80)
    driver_node: int = 0
    driver_first_milestone: float = 0.005
    wave2_jitter: float = 0.35
    wave2_prop_us: float = 500.0

    # Explicit local changes: (absolute time in seconds, node, new value).
    # Only used in explicit diagnostic mode, never the business workload.
    workload: str = "business"
    cpus_per_node: float = 8.0
    memory_per_node_mib: float = 4096.0
    # Business input: legacy five columns, or
    # (task_id, submit_s, node, task_class, cpu, memory_mib, work_cpu_s, duration_s).
    # None generates tasks from the knobs below; () submits no tasks.
    business_tasks: tuple | None = None
    tasks_per_node: int = 3
    task_cpu: float = 4.0
    task_memory_mib: float = 1024.0
    task_duration_s: float = 0.2
    task_interval_s: float = 0.02
    resource_changes: tuple | None = None
    trace_network: bool = False
    message_bytes: int = 970
    frame_overhead_bytes: int = 64
    node_bandwidth_mbps: float = 1000.0
    gcs_bandwidth_mbps: float = 1000.0
    serialize_us_per_byte: float = 0.001

    # ------------------------------------------------------------------ #
    # Bookkeeping / safety limits
    # ------------------------------------------------------------------ #
    max_sim_s: float = 20000.0
    detail_max_nodes: int = 64
    detail_sample_nodes: int = 64

    def __post_init__(self) -> None:
        import math
        if self.workload not in ("business", "diagnostic"):
            raise ValueError("workload must be business or diagnostic")
        if self.workload == "business" and self.resource_changes is not None:
            raise ValueError("resource_changes requires explicit diagnostic workload")
        if not isinstance(self.tasks_per_node, int) or self.tasks_per_node < 0:
            raise ValueError("tasks_per_node must be a nonnegative integer")
        if any(not math.isfinite(v) or v < 0 for v in (self.cpus_per_node,
            self.memory_per_node_mib, self.task_cpu, self.task_memory_mib,
            self.task_duration_s, self.task_interval_s)):
            raise ValueError("invalid business resources or timing")
        if self.node_bandwidth_mbps <= 0 or self.gcs_bandwidth_mbps <= 0:
            raise ValueError("bandwidth must be positive")
        if self.message_bytes <= 0 or self.frame_overhead_bytes < 0 or self.node_proc_us < 0 or self.serialize_us_per_byte < 0:
            raise ValueError("invalid byte size or processing cost")
        if self.num_nodes < 2:
            raise ValueError("num_nodes must be >= 2")
        if self.gcs_write_us <= 0:
            raise ValueError("gcs_write_us must be > 0")


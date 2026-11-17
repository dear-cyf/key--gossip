"""Shared parameters and the message-size model.

Control-variable policy
-----------------------
Everything that is *not* the propagation protocol is taken from the existing
simulations by importing them (never by copying constants):

* resource-update generator (wave 1 + milestone-driven wave 2),
* per-hop network latency (``uplink_us`` / ``prop_us``),
* executor costs (``gcs_fixed_us`` / ``gcs_fanout_us`` / ``gcs_write_us``),
* convergence bookkeeping and the de-duplication rule of
  ``RaySyncerBidiReactorBase::PushToSendingQueue``.

The gossip arm additionally needs a *node-to-node* hop latency.  We reuse the
same one-way link latency as the star's GCS -> Raylet hop (``prop_us``), so the
network parameter itself is shared; the only difference is the number of hops.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # .../try

STAGE1_DIR = os.path.join(REPO, "01_raysyncer_reproduction", "stage1_simulation")
STAGE2_DIR = os.path.join(REPO, "02_batching", "stage2_batching_sim")

for _p in (STAGE1_DIR, STAGE2_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

US = 1e-6  # microseconds -> seconds


def stage1_config_class():
    """``raysyncer_sim.config.SimConfig`` -- pre-batching star baseline."""
    from raysyncer_sim.config import SimConfig  # noqa: WPS433 (late import)

    return SimConfig


def stage2_config_class():
    """``batch_sim.config.SimConfig`` -- adds the PR #57641 batching knobs."""
    from batch_sim.config import SimConfig  # noqa: WPS433 (late import)

    return SimConfig


@dataclass(frozen=True)
class GossipParams:
    """Only the propagation protocol differs from the star baseline."""

    #: number of out-neighbours each Raylet keeps (1 ring edge + k-1 hashed).
    degree_k: int = 4
    #: number of neighbours a node relays a freshly accepted update to.
    fanout_f: int = 2
    #: do not send a relayed message back to the neighbour we got it from.
    exclude_last_hop: bool = True
    #: how the per-message fan-out targets are picked:
    #:   "hash"   -> deterministic ranking by hash(message id, candidate)
    #:   "random" -> per-node RNG draw (same uniform distribution)
    fanout_mode: str = "hash"
    #: how the out-neighbour set is built:
    #:   "rendezvous" -> ring successor + top-(k-1) of a symmetric pair hash
    #:   "random"     -> ring successor + (k-1) uniformly random peers
    neighbor_mode: str = "rendezvous"
    #: seed for the deterministic rendezvous ranking (kept separate from the
    #: workload seed so topology and workload can be varied independently).
    topology_seed: int = 0x5EED1234
    #: one gossip hop = same one-way link cost as the star's GCS -> Raylet hop.
    hop_us: float | None = None
    #: payload bytes of one RaySyncMessage (None -> analytic estimate).
    msg_bytes: int | None = None
    #: per-frame gRPC/HTTP2 overhead, counted once per write event.
    frame_overhead_bytes: int = 64
    #: anti-entropy: period between digest rounds (0 disables it).
    antientropy_period_ms: float = 0.0
    #: digest is sliced over this many rounds (design §6.2).
    antientropy_slices: int = 8
    #: neighbours contacted per digest round (1 = design default, k = all).
    antientropy_peers: int = 1
    #: anti-entropy exchange style:
    #:   "push"     -> the round pushes my newer slice entries to the peer
    #:                 (two independent push rounds per pair, the current sim)
    #:   "pushpull" -> one round trip repairs BOTH sides: my digest goes out,
    #:                 the peer replies with its newer entries (pull) and its
    #:                 requests (push), as designed in 01_design.md §6.2
    ae_mode: str = "push"
    #: bytes of one digest control message (header + 8 B per entry).
    digest_header_bytes: int = 40
    digest_entry_bytes: int = 8
    ae_max_sim_s: float | None = None
    ae_deduplicate_inflight: bool = True
    #: reply with the newer version when a stale duplicate arrives (turns the
    #: redundant push traffic into a repair opportunity).
    repair_on_stale: bool = False

    def resolved_ae_deadline_s(self, cfg) -> float:
        if self.ae_max_sim_s is None:
            return float(cfg.max_sim_s)
        return min(float(self.ae_max_sim_s), float(cfg.max_sim_s))

    def resolved_hop_us(self, cfg) -> float:
        return float(cfg.prop_us if self.hop_us is None else self.hop_us)

    def resolved_msg_bytes(self) -> int:
        return int(
            estimate_message_bytes()["total"] if self.msg_bytes is None
            else self.msg_bytes
        )


# --------------------------------------------------------------------------- #
# Message size
# --------------------------------------------------------------------------- #
def _varint_len(value: int) -> int:
    n = 1
    value >>= 7
    while value:
        n += 1
        value >>= 7
    return n


def _map_entry_bytes(key_len: int, value_bytes: int) -> int:
    """size of one ``map<string, X>`` entry (key field + value field)."""
    inner = 1 + _varint_len(key_len) + key_len + value_bytes
    return 1 + _varint_len(inner) + inner


def _string_field_bytes(length: int) -> int:
    return 1 + _varint_len(length) + length


def estimate_message_bytes(
    std_resources: tuple = (
        "CPU", "GPU", "memory", "object_store_memory",
        "accelerator_type:GeForce-RTX-4050-Laptop-GPU", "node:127.0.0.16",
        "node:__internal_head__",
    ),
    pg_resources: int = 4,
    label_kv: int = 1,
) -> dict:
    """Analytic size of one serialized ``RaySyncMessage`` on the wire.

    Built from the real ``ray_syncer.proto`` field layout and the resource set
    observed on the stage-3 cluster (``workload.json`` -> ``final_nodes[].Resources``):
    7 standard resources plus one ``CPU_group_<32 hex>`` placement-group
    resource per live PG (``pgs_per_node=4``).

    ``map<string,double>`` entry  = tag + varint(len) + [tag + key] + [tag + 8B]
    ``ResourceViewSyncMessage``   = resources_available + resources_total
                                    + flags + node_activity + labels
    ``RaySyncMessage``            = wrapper (version/type/node_id/epoch) + payload
    """
    avail = sum(_map_entry_bytes(len(k), 9) for k in std_resources)
    avail += pg_resources * _map_entry_bytes(42, 9)  # "CPU_group_" + 32 hex chars
    total_map = avail  # resources_total mirrors resources_available's key set
    labels = label_kv * _map_entry_bytes(12, _string_field_bytes(40))
    flags = 2 + 1 + _varint_len(3000) + 2 + _string_field_bytes(12)

    payload = avail + total_map + flags + labels
    sync_field = _string_field_bytes(payload)
    wrapper = 1 + _varint_len(2000) + 2 + sync_field + _string_field_bytes(28) + 1 + 6
    return {
        "resources_available": avail,
        "resources_total": total_map,
        "labels": labels,
        "flags_and_activity": flags,
        "resource_view_payload": payload,
        "sync_message_payload_field": sync_field,
        "wrapper_overhead": wrapper - sync_field,
        "total": wrapper,
    }

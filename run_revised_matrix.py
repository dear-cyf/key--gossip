"""Business-driven resource-sync matrix; uses the updated production kernels."""
from pathlib import Path
import argparse
from array import array
from dataclasses import asdict, replace
import csv
import hashlib
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '03_gossip/gossip_sim'))
from run_gossip_sim import build_arm
from gossip_sim.core import AE_QUEUE, GossipSimulator, RELAY_QUEUE
from gossip_sim.params import GossipParams
from raysyncer_sim import Simulator as Star
from batch_sim import Simulator as Batch

OUT = ROOT / 'outputs/business_revision_20261104/after'


def csv_save(path, rows):
    if rows:
        with path.open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
            w.writeheader()
            w.writerows(rows)


class Recorder:
    trace_nodes = ()

    def __init__(self, n):
        self.versions = np.zeros(n, dtype=np.int64)
        self.view = np.full((n, n), -1, dtype=np.int64)
        self.accept_t = np.full((n, n), np.inf)
        self.emits = {}
        self.latencies = array('d')
        self.seals = []
        self.services = []
        self.source_updates = []

    def on_initial_resources(self, t, states):
        self.view.fill(0)
        self.accept_t.fill(t)

    def on_emit(self, t, msg):
        self.versions[msg.src] = msg.version
        self.view[msg.src, msg.src] = msg.version
        self.accept_t[msg.src, msg.src] = t
        self.emits[msg.src, msg.version] = t
        values = asdict(msg.resources) if hasattr(msg.resources, '__dataclass_fields__') else dict(value=msg.resources)
        self.source_updates.append(dict(time_s=t,node=msg.src,version=msg.version,**values))

    def on_recv(self, t, dst, msg):
        if dst != msg.src and msg.version > self.view[dst, msg.src]:
            self.view[dst, msg.src] = msg.version
            self.accept_t[dst, msg.src] = t
            self.latencies.append((t-self.emits[msg.src, msg.version])*1000)

    def on_seal(self, t, dst, batch, trigger):
        self.seals.append(dict(seal_s=t, dst=dst, size=len(batch), trigger=trigger,
                               oldest_enqueue_s=min(x[1] for x in batch)))

    def on_write_service(self, dst, seal, start, end):
        self.services.append(dict(dst=dst,seal_s=seal,service_start_s=start,service_end_s=end))

    def on_gcs_in(self, *args, **kwargs): pass
    def on_fanout(self, *args): pass
    def on_write(self, *args): pass
    def on_sample(self, *args): pass


class MeasuredGossip(GossipSimulator):
    def _on_recv(self, dst, msg, sender, hop=1, fresh=True):
        # Include all AE lineages, once per accepted remote version.
        self.rec.on_recv(self.now, dst, msg)
        super()._on_recv(dst, msg, sender, hop, fresh)


def percentile(a, q):
    return float(np.percentile(a, q, method='inverted_cdf'))


def run(n, arm, seed, bandwidth, *, gossip_params=None, trace_network=None, **options):
    cfg, _ = build_arm('star_batch500' if arm == 'star_batch1000' else arm, n, seed)
    if arm == 'star_batch1000':
        cfg = replace(cfg, max_batch_delay_ms=1000.)
    cfg = replace(cfg, node_bandwidth_mbps=bandwidth, gcs_bandwidth_mbps=bandwidth,
                  trace_network=arm.startswith("star_batch") if trace_network is None else trace_network, **options)
    rec = Recorder(n)
    gp = gossip_params or GossipParams(antientropy_period_ms=50, antientropy_peers=4)
    sim = (MeasuredGossip(cfg, gp, rec) if arm == 'gossip' else
           (Star if arm == 'star_pre' else Batch)(cfg, rec))
    t = time.perf_counter()
    result = sim.run()
    wall = time.perf_counter()-t
    pairs = ~np.eye(n, dtype=bool)
    covered = (rec.view == rec.versions[None, :]) & (rec.versions[None, :] >= 0)
    times = np.where(covered, rec.accept_t, np.inf)
    node_times = times.max(axis=1)
    lat = np.asarray(rec.latencies)
    business = sim.business.summary() if hasattr(sim, 'business') else None
    start_s = sim.business.start_s if business is not None else cfg.wave1_start_s
    complete = bool(np.all(covered)) and (business is None or business['business_complete'])
    complete = complete and sim.termination_reason != 'max_sim_s'
    row = dict(N=n, protocol=arm, seed=seed, updates=len(rec.emits),
               status='converged' if complete else 'incomplete',
               workload=cfg.workload, measurement_start_s=start_s,
               final_node_coverage=float(np.all(covered, axis=1).mean()),
               final_pair_coverage=float(covered[pairs].mean()))
    for q in (50, 99, 100):
        value = percentile(node_times,q)-start_s
        row[f'T{q}_nodes_s'] = max(0.,value) if complete and np.isfinite(value) else None
        row[f'latency_P{q}_ms'] = percentile(lat, q) if len(lat) else None
    net = sim.net
    row.update(total_messages=int(sum(v[0] for v in net.sent.values())),
               total_network_sends=int(net.stats['frames_sent']),
               total_bytes_est=int(net.stats['bytes_sent']),
               tx_wait_mean_ms=net.stats['tx_wait_s']/max(1, net.stats['frames_sent'])*1000,
               tx_wait_max_ms=net.stats['tx_wait_max_s']*1000,
               rx_wait_max_ms=net.stats['rx_wait_max_s']*1000,
               executor_wait_max_ms=sim.executor_wait_max_s*1000,
               node_processing_wait_max_ms=getattr(sim, 'processing_wait_max_s', 0)*1000,
               outstanding_frames_peak=int(net.stats['outstanding_peak']),
               ae_data_messages=net.kinds['ae_data'][0], digest_messages=net.kinds['digest'][0],
               propagation_messages=net.kinds['relay'][0]+net.kinds['propagation'][0],
               uplink_messages=net.kinds['uplink'][0],
               simulation_end_s=sim.now, wall_clock_s=wall)
    row.update(final_versions_covered_at_s=float(node_times.max()) if np.all(covered) else None,
               termination_reason=sim.termination_reason,
               ae_deadline_s=gp.resolved_ae_deadline_s(cfg) if arm=='gossip' else None)
    checks = dict(bytes_conserved=net.stats['bytes_sent']==net.stats['bytes_received'],
                  frames_conserved=net.stats['frames_sent']==net.stats['frames_received'],
                  queues_drained=net.active==0)
    if arm == 'gossip':
        possible_deliveries = len(rec.emits) * (n - 1)
        row.update(relay_queue=RELAY_QUEUE, relay_overwrites=result.relay_overwrites,
            relay_pending_peak=result.relay_pending_peak,
            relay_inflight_peak=result.relay_inflight_peak,
            relay_pending_at_end=result.relay_pending_at_end,
            relay_inflight_at_end=result.relay_inflight_at_end,
            relay_update_events=result.relay_update_events,
            relay_update_events_completed=result.relay_update_events_completed,
            relay_update_cpu_completed_s=result.relay_update_cpu_completed_s,
            relay_write_cpu_completed_s=result.relay_write_cpu_completed_s,
            ae_queue=AE_QUEUE, ae_repair_overwrites=result.ae_repair_overwrites,
            ae_repairs_suppressed=result.ae_repairs_suppressed,
            ae_requests_suppressed=result.ae_requests_suppressed,
            ae_requests_pruned=result.ae_requests_pruned,
            ae_digest_overwrites=result.ae_digest_overwrites,
            remote_version_acceptance=len(lat)/possible_deliveries if possible_deliveries else None)
        checks['relay_slots_drained'] = all(
            not buffer.pending and not buffer.scheduled and not buffer.inflight
            for peers in sim.relay_buffers for buffer in peers.values())
        checks['relay_updates_drained'] = (
            all(work is None for work in sim.relay_update_work)
            and sim.relay_update_batch == sim.relay_processed_batch
            and result.relay_update_events == result.relay_update_events_completed)
        if checks['relay_slots_drained']:
            checks['relay_attempts_accounted_for'] = result.pushes_attempted == (
                result.relay_msgs + result.relay_overwrites + result.pushes_suppressed_peer_has_it)
        checks['ae_inflight_drained'] = not sim.ae_pending
        checks['ae_requests_drained'] = not sim.ae_requests
        checks['ae_digests_drained'] = not sim.ae_digest_pending
        checks['ae_repair_slots_drained'] = all(
            not buffer.pending and not buffer.scheduled and not buffer.inflight
            for peers in sim.ae_repair_buffers for buffer in peers.values())
        if checks['ae_repair_slots_drained']:
            checks['ae_repairs_accounted_for'] = (
                result.ae_data_msgs == result.ae_data_sent + result.ae_repair_overwrites)
    if business is not None:
        checks['business_queues_drained'] = not any(sim.business.queues) and not any(sim.business.running)
        if arm == 'gossip':
            checks['executors_drained'] = not any(sim.exec_busy) and not any(sim.exec_q)
        else:
            checks['executors_drained'] = not sim.exec_busy and not sim.exec_q
            checks['reactors_drained'] = all(not r.buffer and r.inflight is None for r in sim.reactors)
        row.update(tasks=business['tasks'],tasks_completed=business['completed'],
            cpu_cost_s=business['cpu_cost_s'],memory_cost_mib_s=business['memory_cost_mib_s'],
            requested_work_cpu_s=business['requested_work_cpu_s'],
            tasks_rejected=business['rejected'],tasks_queued=business['queued'],
            tasks_running=business['running'],resource_changes=business['resource_changes'],
            business_complete=business['business_complete'],
            last_business_event_s=business['last_business_event_s'],
            final_version_min=int(rec.versions.min()),final_version_max=int(rec.versions.max()))
        changes = {(x['node'], x['version']):x for x in sim.business.changes}
        checks['every_emit_has_actual_change'] = set(rec.emits)==set(changes)
        checks['causal_timestamps_match'] = all(rec.emits[k]==v['time_s'] for k,v in changes.items())
        checks['state_actually_changed'] = all(x['cpu_before']!=x['cpu_after'] or
            x['memory_before_mib']!=x['memory_after_mib'] for x in changes.values())
        checks['per_node_versions_match_changes'] = all(int(rec.versions[i])==sum(
            x['node']==i for x in sim.business.changes) for i in range(n))
        checks['final_payloads_match_versions'] = all(
            not covered[d,src] or sim.replica_resources[d,src]==sim.resource_history[src,int(rec.versions[src])]
            for d in range(n) for src in range(n))
        if business['business_complete']:
            checks['all_resources_released'] = business['all_resources_released']
    else:
        row.update(tasks=0,tasks_completed=0,tasks_rejected=0,tasks_queued=0,tasks_running=0,
            resource_changes=len(rec.emits),business_complete=True,last_business_event_s=sim.workload_last_s,
            final_version_min=int(rec.versions.min()),final_version_max=int(rec.versions.max()))
    if rec.seals:
        row['batches'] = len(rec.seals)
        row['batch_size_mean'] = float(np.mean([x['size'] for x in rec.seals]))
        checks['one_entry_per_remote_source'] = (not cfg.gcs_reactor_dedup or
            all(x['size']<=n-1 for x in rec.seals))
        checks['pr_send_triggers'] = all(x['trigger'] in ('size','timer','write_done','immediate') for x in rec.seals)
        # Every seal owns the stream until its abstract write service completes.
        services = {(x['dst'], x['seal_s']):x for x in rec.services}
        previous_end = {}
        checks['one_active_write_per_stream'] = True
        for seal in rec.seals:
            dst, t = seal['dst'], seal['seal_s']
            checks['one_active_write_per_stream'] &= t >= previous_end.get(dst, -float('inf'))
            service = services.get((dst,t))
            previous_end[dst] = service['service_end_s'] if service else float('inf')
    else:
        row.update(batches=0, batch_size_mean=1.)
    # At a cutoff, outstanding network work is reported, not mislabeled as a bug.
    invariant_checks = {k:v for k,v in checks.items() if k not in
                        ('bytes_conserved','frames_conserved','queues_drained',
                         'business_queues_drained','executors_drained','ae_inflight_drained',
                         'ae_requests_drained','ae_digests_drained','ae_repair_slots_drained',
                         'reactors_drained','relay_slots_drained','relay_updates_drained') or complete}
    assert all(invariant_checks.values()), invariant_checks
    folder = OUT / f'{arm}_N{n}_seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'run.json').write_text(json.dumps(dict(metrics=row, checks=checks,
        config=asdict(cfg), gossip_params=asdict(gp) if arm=='gossip' else None,
        network_stats=dict(net.stats),business=business), indent=2), encoding='utf-8')
    csv_save(folder/'source_updates.csv', rec.source_updates)
    if business is not None:
        csv_save(folder/'tasks.csv', [asdict(task) for task in sim.business.tasks])
        csv_save(folder/'business_events.csv', sim.business.events)
        csv_save(folder/'resource_changes.csv', sim.business.changes)
        csv_save(folder/'resource_observations.csv', sim.business.observations)
    csv_save(folder/'batches.csv', rec.seals)
    csv_save(folder/'write_service.csv', rec.services)
    csv_save(folder/'network_trace.csv', net.trace)
    csv_save(folder/'node_load.csv', [dict(node=i, sent_messages=net.sent[i][0],
        sent_frames=net.sent[i][1], sent_bytes=net.sent[i][2],
        received_messages=net.received[i][0],received_frames=net.received[i][1],
        received_bytes=net.received[i][2]) for i in range(n+1)])
    sorted_nodes = np.sort(node_times)
    sorted_pairs = np.sort(times[pairs])
    ticks = np.arange(0, min(sim.now,float(np.max(node_times)))+.1, .1)
    csv_save(folder/'coverage.csv', [dict(time_s=float(t),
        final_version_node_coverage=float(np.searchsorted(sorted_nodes,t,side='right')/n),
        final_version_pair_coverage=float(np.searchsorted(sorted_pairs,t,side='right')/(n*(n-1)))) for t in ticks])
    csv_save(folder/'traffic_100ms.csv', [dict(window_start_s=k*.1,messages=v[0],frames=v[1],bytes=v[2]) for k,v in sorted(net.bins.items())])
    np.savez_compressed(folder/'final_state.npz', versions=rec.versions, received_versions=rec.view,
                        final_accept_times_s=rec.accept_t, accepted_latencies_ms=lat)
    print(json.dumps(row), flush=True)
    return row


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--nodes', nargs='+',type=int,default=[100,300,1000])
    ap.add_argument('--arms',nargs='+',default=['star_pre','star_batch500','star_batch1000','gossip'])
    ap.add_argument('--seeds',nargs='+',type=int,default=[20251011])
    ap.add_argument('--bandwidth-mbps',type=float,default=1000.)
    ap.add_argument('--out',type=Path,default=OUT)
    ap.add_argument('--tasks-per-node',type=int,default=3)
    ap.add_argument('--task-cpu',type=float,default=4.)
    ap.add_argument('--task-memory-mib',type=float,default=1024.)
    ap.add_argument('--task-duration-s',type=float,default=.2)
    ap.add_argument('--task-interval-s',type=float,default=.02)
    ap.add_argument('--cpus-per-node',type=float,default=8.)
    ap.add_argument('--memory-per-node-mib',type=float,default=4096.)
    ap.add_argument('--tasks-json',type=Path,help='JSON array of eight-column task rows (legacy five-column rows also accepted)')
    ap.add_argument('--max-sim-s',type=float,default=20000.)
    args=ap.parse_args()
    OUT=args.out
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in args.seeds:
        for n in args.nodes:
            for arm in args.arms:
                print(f'RUN {n} {arm} seed={seed}',flush=True)
                rows.append(run(n,arm,seed,args.bandwidth_mbps,
                    tasks_per_node=args.tasks_per_node, task_cpu=args.task_cpu,
                    task_memory_mib=args.task_memory_mib, task_duration_s=args.task_duration_s,
                    task_interval_s=args.task_interval_s,cpus_per_node=args.cpus_per_node,
                    memory_per_node_mib=args.memory_per_node_mib,max_sim_s=args.max_sim_s,
                    business_tasks=tuple(tuple(x) for x in json.loads(args.tasks_json.read_text(encoding='utf-8-sig'))) if args.tasks_json else None))
                csv_save(OUT/'summary.csv',rows)
    (OUT/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    paths=[ROOT/'run_revised_matrix.py']
    for folder in ('01_raysyncer_reproduction/stage1_simulation/raysyncer_sim',
                   '02_batching/stage2_batching_sim/batch_sim','03_gossip/gossip_sim/gossip_sim'):
        paths.extend((ROOT/folder).glob('*.py'))
    (OUT/'source_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2),encoding='utf-8')


if __name__=='__main__': main()

"""Seeded Poisson arrivals, saved once and replayed across propagation arms."""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

import run_revised_matrix as matrix
from gossip_sim.params import GossipParams
from raysyncer_sim.business import normalize_task_spec

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / 'outputs/random_heterogeneous_20261105'
SEEDS = [20261104, 20261105, 20261106, 20261107, 20261108]
TASK_COLUMNS = ('task_id', 'submit_s', 'node', 'task_class', 'cpu',
                'memory_mib', 'work_cpu_s', 'duration_s')
TASK_CLASSES = (
    ('small', .4, 2., 512., .2),
    ('medium', .3, 4., 1024., .8),
    ('cpu_large', .2, 8., 2048., 2.),
    ('memory_large', .1, 2., 3072., .8),
)


def matches_relay_queue(arm, saved):
    if arm != 'gossip':
        return True
    metrics = saved['metrics']
    return (metrics.get('relay_queue') == matrix.RELAY_QUEUE
            and metrics.get('ae_queue') == matrix.AE_QUEUE)


def workload_settings():
    return dict(task_columns=TASK_COLUMNS,
        task_classes=[dict(task_class=k, probability=p, cpu=c, memory_mib=m,
                           base_work_cpu_s=w) for k,p,c,m,w in TASK_CLASSES],
        work_multiplier_uniform=[.8,1.2], runtime_formula='work_cpu_s / cpu',
        cpu_cost_formula='cpu * elapsed_runtime_s',
        memory_cost_formula='memory_mib * elapsed_runtime_s',
        cost_scope='logical resource reservation; network costs use sync message bytes')


def save_task_table(path, specs):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(TASK_COLUMNS)
        writer.writerows(normalize_task_spec(s,i) for i,s in enumerate(specs))


def generate_tasks(n, seed, rate=5., start=.5, duration=10., cpu=None, memory=None, runtime=None):
    """Poisson arrivals and independent class/work streams; explicit sizes replay legacy inputs."""
    if n < 2 or any(not math.isfinite(x) or x < 0 for x in
                      (rate, start, duration)):
        raise ValueError('invalid workload parameters')
    legacy = any(v is not None for v in (cpu, memory, runtime))
    if legacy:
        cpu = 4. if cpu is None else cpu
        memory = 1024. if memory is None else memory
        runtime = .2 if runtime is None else runtime
        if any(not math.isfinite(v) or v < 0 for v in (cpu,memory,runtime)):
            raise ValueError('invalid workload parameters')
    if rate == 0 or duration == 0:
        return ()
    # Independent streams keep arrival gaps independent of node marks.
    arrivals = random.Random(seed)
    targets = random.Random(seed ^ 0x554E49464F524D)
    kinds = random.Random(seed ^ 0x434C415353)
    works = random.Random(seed ^ 0x574F524B)
    t = start
    tasks = []
    while True:
        t += arrivals.expovariate(n * rate)
        if t >= start + duration:
            return tuple(tasks)
        node = targets.randrange(n)
        if legacy:
            tasks.append((t, node, cpu, memory, runtime))
            continue
        draw = kinds.random()
        cumulative = 0.
        for kind, probability, c, m, base in TASK_CLASSES:
            cumulative += probability
            if draw < cumulative:
                break
        work = base * works.uniform(.8, 1.2)
        tasks.append((len(tasks), t, node, kind, c, m, work, work/c))


def audit_run(folder, specs, task_hash):
    """Independently replay FIFO and reservations from saved execution timestamps."""
    import csv
    with (folder / 'tasks.csv').open(encoding='utf-8-sig', newline='') as f:
        tasks = list(csv.DictReader(f))
    checks = dict(task_list_replayed_exactly=True, runtime_after_allocation=True,
                  fifo_order=True, resource_conservation=True,
                  work_conserving_fifo=True, allocation_release_versions=True,
                  task_work_runtime_relation=True, resource_time_costs=True)
    by_node = {}
    waits = []
    for tid, (task, spec) in enumerate(zip(tasks, specs)):
        actual = (float(task['submit_s']), int(task['node']), float(task['cpu']),
                  float(task['memory_mib']), float(task['duration_s']))
        normalized = normalize_task_spec(spec, tid)
        checks['task_list_replayed_exactly'] &= actual == (normalized[1],normalized[2],normalized[4],normalized[5],normalized[7])
        if len(spec) == 8:
            checks['task_list_replayed_exactly'] &= (int(task['task_id']), task['task_class'],
                float(task['work_cpu_s'])) == (normalized[0],normalized[3],normalized[6])
            checks['task_work_runtime_relation'] &= math.isclose(normalized[6], actual[2]*actual[4], rel_tol=1e-12)
        checks['runtime_after_allocation'] &= task['status'] == 'completed'
        if task['status'] != 'completed':
            continue
        submit, node, cpu, memory, runtime = actual
        start, finish = float(task['start_s']), float(task['finish_s'])
        checks['runtime_after_allocation'] &= start >= submit and math.isclose(finish-start, runtime, abs_tol=1e-10)
        if 'cpu_cost_s' in task:
            checks['resource_time_costs'] &= (math.isclose(float(task['cpu_cost_s']),cpu*runtime,abs_tol=1e-10)
                and math.isclose(float(task['memory_cost_mib_s']),memory*runtime,abs_tol=1e-8))
        waits.append(start-submit)
        by_node.setdefault(node, []).append((submit, int(task['task_id']), start, finish, cpu, memory))
    checks['task_list_replayed_exactly'] &= len(tasks) == len(specs)
    config = json.loads((folder/'run.json').read_text())['config']
    for entries in by_node.values():
        entries.sort()
        checks['fifo_order'] &= all(a[2] <= b[2] for a,b in zip(entries, entries[1:]))
        # Independent scheduling oracle, with finishes processed at their actual
        # times and CPU/memory both checked for admission of the FIFO head.
        running = []
        previous_start = 0.
        for submit, tid, start, finish, cpu, memory in entries:
            expected = max(submit, previous_start)
            while True:
                running = [r for r in running if r[0] > expected + 1e-12]
                if (sum(r[1] for r in running)+cpu <= config['cpus_per_node']+1e-10
                        and sum(r[2] for r in running)+memory <= config['memory_per_node_mib']+1e-10):
                    break
                expected = min(r[0] for r in running)
            checks['work_conserving_fifo'] &= math.isclose(start, expected, abs_tol=1e-10)
            running.append((finish, cpu, memory))
            previous_start = start
        points = []
        for _, tid, start, finish, cpu, memory in entries:
            points.extend([(start,1,tid,cpu,memory), (finish,0,tid,-cpu,-memory)])
        used_cpu = used_memory = 0.
        for _,_,_,cpu,memory in sorted(points):
            used_cpu += cpu
            used_memory += memory
            checks['resource_conservation'] &= (-1e-10 <= used_cpu <= config['cpus_per_node']+1e-10
                and -1e-10 <= used_memory <= config['memory_per_node_mib']+1e-10)
    with (folder/'resource_changes.csv').open(encoding='utf-8-sig', newline='') as f:
        changes = list(csv.DictReader(f))
    versions = {}
    causes = {}
    for change in changes:
        node, tid = int(change['node']), int(change['task_id'])
        versions[node] = versions.get(node,0)+1
        checks['allocation_release_versions'] &= int(change['version']) == versions[node]
        causes.setdefault(tid, []).append(change['cause'])
    checks['allocation_release_versions'] &= len(changes) == 2*len(specs) and all(
        causes.get(tid) == ['allocate','release'] for tid in range(len(specs)))
    stats = dict(task_list_sha256=task_hash, tasks_ever_waited=sum(w>1e-10 for w in waits),
                 queue_wait_mean_s=statistics.mean(waits) if waits else 0.,
                 queue_wait_max_s=max(waits, default=0.))
    (folder/'workload_audit.json').write_text(json.dumps(dict(checks=checks, metrics=stats),indent=2),encoding='utf-8')
    assert all(checks.values()), (str(folder), checks)
    return stats


def write_report(out, rows):
    lines = ['# 随机任务模型调试结果', '',
        '全局泊松率为 N×5/s，每次到达独立均匀选择目标节点，等价于每节点 5/s 的独立泊松流。提交窗口 [0.5, 10.5) s。',
        '每节点 8 CPU、4096 MiB；小/中/CPU大/大内存任务按 40%/30%/20%/10% 抽样。计算量为各类基准乘 U(0.8,1.2)，运行时间=计算量/CPU；理论 CPU 提供负载约 50%。业务成本按 CPU×已运行时间及 MiB×已运行时间计，网络成本仍按同步消息字节计。',
        '资源不足按节点 FIFO 排队。分配与释放分别产生版本；排队和远端接收不产生本地版本。初始版本 0 全员已知。',
        '各协议重放同一份已保存任务清单（SHA-256 可核对）。保留既有链路、处理成本和 Gossip 拓扑参数；AE 周期 50 ms，每轮 4 个邻居、8 个切片、push 模式。',
        '普通 Gossip 和 AE 补发分别按目标邻居和源节点保留最新待发快照。AE 对未完成的同版本请求去重，在写入前移除已满足条目，并按目标和分片合并待发摘要。逐次结果记录覆盖和去重次数。',
        'AE 的独立截止参数 ae_max_sim_s=null，表示受实验 max_sim_s=60 s 约束；不会按额外固定时限停止修复。业务完成且最终版本全部覆盖后，停止产生新 AE 轮次，并排空在途消息和执行队列。',
        '覆盖时间从最终版本接收矩阵计算，独立于排空时间；中间版本允许被合并，延迟分位数仅统计实际接受的远端版本。', '',
        '| N | 协议 | 种子 | 任务数 | 等待任务 | 最终覆盖时刻/s | 仿真结束/s | 状态 |',
        '|---:|---|---:|---:|---:|---:|---:|---|']
    for r in rows:
        cov = r['final_versions_covered_at_s']
        lines.append(f"| {r['N']} | {r['protocol']} | {r['seed']} | {r['tasks']} | {r['tasks_ever_waited']} | {cov if cov is not None else '未覆盖'} | {r['simulation_end_s']:.6f} | {r['status']} |")
    lines.extend(['', '校验包括：精确任务重放、运行计时、FIFO、独立调度重算、资源守恒、每任务两次资源版本变更、消息/字节守恒、最终资源快照一致、业务/网络/执行/AE 队列排空。详见各运行目录 run.json 与 workload_audit.json。',
                  '原始任务在 workloads/；summary.csv 为逐次结果，summary.json 为完整汇总。未做真实集群性能校准。'])
    (out/'报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def execute_job(job):
    out,n,seed,arm,replay,digest,max_sim,ae_max=job
    matrix.OUT=out
    print(f'RUN N={n} {arm} seed={seed} tasks={len(replay)}',flush=True)
    row=matrix.run(n,arm,seed,1000.,business_tasks=replay,max_sim_s=max_sim,
        trace_network=False,gossip_params=GossipParams(antientropy_period_ms=50,
            antientropy_peers=4,ae_max_sim_s=ae_max))
    row.update(audit_run(out/f'{arm}_N{n}_seed{seed}',replay,digest))
    save_task_table(out/f'{arm}_N{n}_seed{seed}'/'task_table.csv', replay)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', nargs='+', type=int, default=[100,300])
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--arms', nargs='+', default=['star_pre','star_batch500','star_batch1000','gossip'])
    parser.add_argument('--max-sim-s', type=float, default=60.)
    parser.add_argument('--ae-max-time-s', type=float, default=None,
                        help='independent absolute AE deadline; omitted means experiment limit')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--workers', type=int, default=1,
                        help='independent simulation processes; each run remains sequential')
    parser.add_argument('--resume', action='store_true',help='reuse matching, audited completed runs')
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True,exist_ok=True)
    (out/'workloads').mkdir(exist_ok=True)
    matrix.OUT = out
    config = dict(cluster=dict(num_nodes=args.nodes,cpu_per_node=8,memory_per_node_mib=4096),
        workload=dict(arrival_process='poisson',arrival_rate_per_node_per_s=5,
            target_node_selection='uniform_random',submission_start_s=.5,submission_duration_s=10,
            **workload_settings(),insufficient_resources='queue',
            queue_policy='fifo',runtime_starts_at='resource_allocation'),
        resource_updates=dict(trigger='actual_local_resource_change',increment_version_on_allocation=True,
            increment_version_on_release=True,increment_version_on_queueing=False,increment_version_on_remote_receive=False),
        experiment=dict(max_sim_s=args.max_sim_s,stop_when='workload_complete_and_final_versions_covered_and_queues_drained',
            record_convergence_before_queue_drain=True,workload_seeds=args.seeds,save_task_list=True,
            replay_same_task_list_across_protocols=True),
        anti_entropy=dict(period_ms=50,max_time_s=args.ae_max_time_s),protocols=args.arms)
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    rows=[]
    jobs=[]
    started = time.perf_counter()
    for n in args.nodes:
        for seed in args.seeds:
            specs = generate_tasks(n,seed)
            payload = json.dumps(specs,separators=(',',':')).encode()
            digest = hashlib.sha256(payload).hexdigest()
            path = out/'workloads'/f'N{n}_seed{seed}.json'
            path.write_bytes(payload)
            save_task_table(path.with_suffix('.csv'), specs)
            # Read back the saved input used by every arm, without regeneration.
            replay = tuple(tuple(t) for t in json.loads(path.read_bytes()))
            for arm in args.arms:
                folder=out/f'{arm}_N{n}_seed{seed}'
                if args.resume and (folder/'run.json').exists() and (folder/'workload_audit.json').exists():
                    saved=json.loads((folder/'run.json').read_text())
                    audited=json.loads((folder/'workload_audit.json').read_text())
                    cfg=saved['config']
                    same=(tuple(tuple(t) for t in cfg['business_tasks'])==replay
                        and cfg['max_sim_s']==args.max_sim_s
                        and matches_relay_queue(arm,saved)
                        and (arm!='gossip' or saved['gossip_params']==matrix.asdict(GossipParams(
                            antientropy_period_ms=50,antientropy_peers=4,ae_max_sim_s=args.ae_max_time_s)))
                        and audited['metrics']['task_list_sha256']==digest)
                    if same and all(saved['checks'].values()) and all(audited['checks'].values()):
                        row=saved['metrics']
                        row.update(audited['metrics'])
                        rows.append(row)
                        print(f'REUSE N={n} {arm} seed={seed}',flush=True)
                        continue
                jobs.append((out,n,seed,arm,replay,digest,args.max_sim_s,args.ae_max_time_s))
    def save_rows():
        rows.sort(key=lambda r:(r['N'],r['seed'],args.arms.index(r['protocol'])))
        matrix.csv_save(out/'summary.csv',rows)
        (out/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        write_report(out,rows)
    save_rows()
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.workers==1:
        for job in jobs:
            rows.append(execute_job(job))
            save_rows()
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(execute_job,job) for job in jobs]
            for future in as_completed(futures):
                rows.append(future.result())
                save_rows()
    paths=[ROOT/'run_random_matrix.py',ROOT/'run_revised_matrix.py']
    for directory in ('01_raysyncer_reproduction/stage1_simulation/raysyncer_sim',
                      '02_batching/stage2_batching_sim/batch_sim','03_gossip/gossip_sim/gossip_sim'):
        paths.extend((ROOT/directory).glob('*.py'))
    (out/'source_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths},indent=2),encoding='utf-8')
    print(f'COMPLETE {len(rows)} runs, wall={time.perf_counter()-started:.1f}s',flush=True)


if __name__ == '__main__':
    main()

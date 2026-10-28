"""Process-isolated parallel evaluation, with immutable manifests and no retries."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from .protocol import load_protocol, variants, sha256, atomic_json, read_json, source_hashes


def worker(protocol_path, variant_id, device):
    import torch
    from .runner import build_runtime, run_case
    protocol = load_protocol(protocol_path)
    protocol['_sha256'] = sha256(protocol_path)
    variant = next(v for v in variants(protocol) if v['id'] == variant_id)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    runtime, model, metadata = build_runtime(variant, device)
    root = Path(protocol['output_root']) / 'runs' / variant_id
    root.mkdir(parents=True, exist_ok=True)
    manifest = dict(variant=variant, **metadata, protocol_sha256=protocol['_sha256'])
    path = root / 'manifest.json'
    if path.exists() and read_json(path) != manifest:
        raise ValueError('Worker manifest changed; use a new campaign output directory')
    atomic_json(path, manifest)
    outcomes = []
    # Cover every initial condition at short lead before extending the same
    # full-state trajectories. This is scheduling, not a state reinitialization.
    for horizon in sorted(set([min(10,max(protocol['horizons_days'])), *protocol['horizons_days']])):
        outcomes = []
        for case in protocol['cases']:
            result = run_case(protocol, variant, case, runtime, model, metadata, stop_after=horizon)
            outcomes.append(result['status'])
            print(variant_id, case, result['status'], result['completed_days'], flush=True)
    atomic_json(root / 'FINISHED.json', dict(status='finished', cases=len(outcomes),
                completed=outcomes.count('complete'), numerical_failures=outcomes.count('numerical_failure')))


def launch(protocol_path, gpus):
    import fcntl
    protocol = load_protocol(protocol_path)
    root = Path(protocol['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'scheduler.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    digest = sha256(protocol_path)
    source = Path(__file__).resolve().parents[2]
    identity = dict(protocol_sha256=digest, protocol=protocol, source_hashes=source_hashes(source))
    manifest_path = root / 'campaign.json'
    if manifest_path.exists() and read_json(manifest_path) != identity:
        raise ValueError('Protocol/source changed since launch; create a new campaign instead')
    atomic_json(manifest_path, identity)
    children = []
    logs = root / 'logs'
    logs.mkdir(exist_ok=True)
    for i, variant in enumerate(variants(protocol)):
        reuse=protocol.get('reuse')
        if reuse and variant['policy'] in reuse['policies']:
            target=Path(reuse['root'])/'runs'/variant['id']
            existing=read_json(target/'manifest.json')
            if existing['variant'] != variant:
                raise ValueError('Reused variant differs from registered control')
            link=root/'runs'/variant['id'];link.parent.mkdir(exist_ok=True)
            if not link.exists():link.symlink_to(target,target_is_directory=True)
            if not link.is_symlink() or link.resolve()!=target.resolve():
                raise ValueError('Reused results must be an explicit link to the parent campaign')
            continue
        if (root / 'runs' / variant['id'] / 'FINISHED.json').exists():
            continue
        gpu = gpus[i % len(gpus)]
        stream = (logs / f"{variant['id']}.log").open('a')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        args = [sys.executable, '-m', 'neuralpom.evaluation.campaign', 'worker',
                '--protocol', str(Path(protocol_path).resolve()), '--variant', variant['id'], '--device', 'cuda:0']
        process = subprocess.Popen(args, stdout=stream, stderr=subprocess.STDOUT, env=env)
        children.append((process, stream, variant['id'], gpu))
    while True:
        states = [dict(variant=name, gpu=gpu, pid=p.pid, returncode=p.poll()) for p, _, name, gpu in children]
        atomic_json(root / 'scheduler.json', dict(pid=os.getpid(), updated_at=time.time(), workers=states,
                    state='running' if any(s['returncode'] is None for s in states) else 'finished'))
        if all(s['returncode'] is not None for s in states):
            break
        time.sleep(10)
    for _, stream, _, _ in children:
        stream.close()
    if any(s['returncode'] != 0 for s in states):
        raise RuntimeError('Some workers failed; inspect scheduler.json and per-variant logs; no automatic retry')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    start = sub.add_parser('launch')
    start.add_argument('--protocol', required=True)
    start.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    run = sub.add_parser('worker')
    run.add_argument('--protocol', required=True)
    run.add_argument('--variant', required=True)
    run.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.action == 'launch':
        launch(args.protocol, args.gpus.split(','))
    else:
        worker(args.protocol, args.variant, args.device)


if __name__ == '__main__':
    main()

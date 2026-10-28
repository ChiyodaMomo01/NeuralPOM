"""One frozen trajectory; training provenance and inference policy stay distinct."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from neuralpom.models.corrector_net import CorrectorNet
from neuralpom.v1.config import load_config
from neuralpom.v1.runtime import Runtime
from neuralpom.v1.diagnostics import state_diagnostics, numpy_fields
from neuralpom.v1.checkpoint import capture, restore, atomic_torch
from neuralpom.hybrid.sparse_physics import SparsePhysics
from .protocol import FIELDS, atomic_json, read_json, sha256, case_root
from .correction import correct


BARE = dict(mode='eager', cg_backend='torch', compiled_kernels=[], cuda_graph=False,
            checkpoint_physics=False, torch_compile=False, autocast=False,
            inference_fast_paths=False)


def configure_bare(runtime):
    # Calibration is checked against the actual training arithmetic first.
    # The separately recorded execution policy changes only implementation.
    runtime.physics = SparsePhysics(runtime.device, acceleration='eager', cg_backend='torch',
        compiled_kernels=(), checkpoint_physics=False, core_name=runtime.spec.name)
    runtime.physics._configure(runtime.core)
    runtime.baseline = capture(runtime.core, cpu=False)
    return dict(BARE)


def execution_audit(runtime):
    core = runtime.core
    flags = ('use_triton_elliptic_operator', 'use_triton_cg_updates', 'compile_eta_solver',
             'generation_fast_path', 'runtime_no_grad_fast_path', 'runtime_cuda_graph_fast_path',
             'runtime_vamax_safe_fast_path', 'runtime_ramp_saturated_fast_path')
    observed = {name: bool(getattr(core, name, False)) for name in flags}
    if any(observed.values()) or runtime.physics.acceleration != 'eager' or runtime.physics.compiled_kernels:
        raise ValueError(f'An inference optimization is enabled: {observed}')
    return dict(BARE, flags=observed, core_class=type(core).__module__ + '.' + type(core).__name__)


def build_runtime(variant, device):
    if variant.get("model_family") == "ai":
        from neuralpom.ai_baselines.evaluation import build_runtime as build_ai
        return build_ai(variant, device)
    config = load_config(variant['config'])
    if (config['core'], config['frequency']) != (variant['core'], variant['frequency']):
        raise ValueError('Registered core/frequency differs from training configuration')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    runtime = Runtime(config, device)
    contract = runtime.contract()
    path = Path(config['run_root']) / 'frozen' / f"{variant['stage']}.pt"
    if variant['stage'] == 'physics':
        model, metadata = None, dict(step=None, selection=None, weight_sha256=None)
    else:
        value = torch.load(path, map_location='cpu', weights_only=False)
        if value['contract'] != contract or value['stage'] != variant['stage']:
            raise ValueError(f'Weight, core, data, calibration, or training contract mismatch: {path}')
        model = CorrectorNet(101, 60, config['model']['hidden_dim']).to(device)
        model.load_state_dict(value['model'], strict=True)
        model.eval().requires_grad_(False)
        metadata = {key: value.get(key) for key in ('step', 'selection', 'posttrain_gradient_mode')}
        metadata.update(weights=str(path), weight_sha256=sha256(path))
    configure_bare(runtime)
    metadata.update(training_contract=contract, inference=execution_audit(runtime),
                    assets_sha256={key: sha256(config[key]) for key in ('input_stats', 'scales', 'front_scales')},
                    config_sha256=sha256(variant['config']), correction_policy=variant['policy'])
    if variant['policy'] == 'density75':
        metadata['policy_details'] = dict(density_gain=.75, neutral_gain=1., cap=5.,
            spatial_filter=False, version='retained-density75-full-spectrum-layer-budget-v1',
            historical_confirmation='implicit_p0_fixed/1h/posttrain; 14 temporal origins')
    return runtime, model, metadata


def save_fields(path, values, model_time):
    path = Path(path)
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **numpy_fields(values), time=np.float64(model_time))
    temporary.replace(path)


def read_records(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def append(path, record):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')


def verify_frozen(model, before):
    if model is not None and any(not torch.equal(value.detach().cpu(), before[key])
                                 for key, value in model.state_dict().items()):
        raise RuntimeError('Frozen inference changed network weights or buffers')


@torch.no_grad()
def advance_interval(runtime):
    try:
        runtime.advance(False)
    except ValueError:
        # Explicit's historical loop returns early when its native velocity
        # guard fires. Distinguish that numerical stop from an I/O/clock bug.
        core = runtime.core
        maximum = float(core.vaf.abs().max())
        if maximum > core.vmaxl:
            raise FloatingPointError(f'Native barotropic velocity limit: {maximum} > {core.vmaxl}')
        raise
    if not math.isfinite(float(runtime.core.time)):
        raise FloatingPointError('Nonfinite physical clock')
    # Detect invalid prognostic state before a network can consume it.
    for name in (*FIELDS, 'et', 'rho', 'q2', 'q2l'):
        if not bool(torch.isfinite(getattr(runtime.core, name)).all()):
            raise FloatingPointError(f'Nonfinite physical state: {name}')
    if float(runtime.core.vaf.abs().max()) > runtime.core.vmaxl:
        raise FloatingPointError('Native barotropic velocity limit exceeded')


@torch.no_grad()
def run_case(protocol, variant, case, runtime, model, metadata, stop_after=None):
    if variant.get("model_family") == "ai":
        from neuralpom.ai_baselines.evaluation import run_case as run_ai
        return run_ai(protocol, variant, case, runtime, model, metadata, stop_after)
    output = case_root(protocol['output_root'], variant['id'], case)
    output.mkdir(parents=True, exist_ok=True)
    target = max(protocol['horizons_days'])
    phase_target = target if stop_after is None else min(int(stop_after), target)
    identity = {'variant': variant['id'], **case, 'target_days': target,
                'weight_sha256': metadata['weight_sha256'], 'protocol_sha256': protocol['_sha256']}
    status_path = output / 'status.json'
    if status_path.exists():
        old = read_json(status_path)
        if any(old.get(k) != v for k, v in identity.items()):
            raise ValueError('Refusing to mix results from different protocol or weights')
        if old['status'] in ('complete', 'numerical_failure'):
            return old
        if old['completed_days'] >= phase_target:
            return old
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()} if model else {}
    runtime.reset(case['member'], case['start_day'])
    execution_audit(runtime)
    geometry = {k: getattr(runtime.core, k) for k in ('fsm', 'dum', 'dvm', 'h', 'art', 'dx', 'dy', 'dz', 'zz')}
    save_fields(output / 'geometry.npz', geometry, case['start_day'])
    save_fields(output / 'initial.npz', {k: getattr(runtime.core, k) for k in FIELDS}, case['start_day'])
    status = dict(identity, status='running', completed_days=0, pid=os.getpid(), updated_at=time.time())
    restart = output / 'restart.pt'
    if restart.exists():
        saved = torch.load(restart, map_location='cpu', weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('Restart provenance differs')
        restore(runtime.core, saved['state'])
        status['completed_days'] = saved['day']
        runtime.assert_time(case['start_day'] + saved['day'])
    # Restart is the commit point. Discard only uncommitted trailing records.
    for name in ('daily', 'spectra', 'profiles'):
        path = output / f'{name}.jsonl'
        rows = [r for r in read_records(path) if r['lead_day'] <= status['completed_days']]
        if [r['lead_day'] for r in rows] != list(range(1, status['completed_days'] + 1)):
            raise ValueError(f'Incomplete committed {name} records')
        path.write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in rows))
    atomic_json(status_path, status)
    try:
        for day in range(status['completed_days'] + 1, phase_target + 1):
            started = time.monotonic()
            limit_factors = []
            for _ in range(86400 // runtime.interval_seconds):
                advance_interval(runtime)
                if model is not None:
                    diagnostics = correct(runtime, model, variant['policy'])
                    limit_factors.append(diagnostics['ts_limit_factor'])
                    for field in (*FIELDS, 'rho'):
                        if not bool(torch.isfinite(getattr(runtime.core, field)).all()):
                            raise FloatingPointError(f'Nonfinite corrected state: {field}')
            runtime.assert_time(case['start_day'] + day)
            reference = runtime.data.load_initial(case['member'], case['start_day'] + day,
                                                  map_location=runtime.device)
            runtime.assert_time(float(reference['time']))
            prediction = {key: getattr(runtime.core, key) for key in FIELDS}
            metrics, profiles, spectra = state_diagnostics(runtime.core, prediction, reference)
            if not all(math.isfinite(v) for v in metrics.values()):
                raise FloatingPointError('Nonfinite diagnostic')
            save_fields(output / f'day{day:03d}_post.npz', prediction, float(runtime.core.time))
            save_fields(output / f'day{day:03d}_reference.npz', {k: reference[k] for k in FIELDS}, float(reference['time']))
            append(output / 'daily.jsonl', dict(lead_day=day, time=float(runtime.core.time),
                reference_time=float(reference['time']), metrics=metrics, wall_seconds=time.monotonic() - started,
                correction_policy=variant['policy'], min_ts_limit_factor=min(limit_factors, default=1.)))
            append(output / 'spectra.jsonl', dict(lead_day=day, **spectra))
            append(output / 'profiles.jsonl', dict(lead_day=day, **profiles))
            atomic_torch(restart, dict(day=day, identity=identity, state=runtime.snapshot()))
            status.update(completed_days=day, updated_at=time.time())
            atomic_json(status_path, status)
        status['status'] = 'complete' if status['completed_days'] == target else 'running'
    except FloatingPointError as error:
        actual = float(runtime.core.time)
        status.update(status='numerical_failure', error=str(error),
                      first_anomaly_day=status['completed_days'] + 1,
                      first_anomaly_time=actual if math.isfinite(actual) else None)
    except Exception as error:
        status.update(status='infrastructure_error', error=f'{type(error).__name__}: {error}',
                      traceback=traceback.format_exc())
        raise
    finally:
        try:
            verify_frozen(model, before)
            status['inference'] = execution_audit(runtime)
        except Exception as error:
            status.update(status='infrastructure_error', error=f'Final integrity check: {error}')
            raise
        finally:
            status['updated_at'] = time.time()
            atomic_json(status_path, status)
    return status

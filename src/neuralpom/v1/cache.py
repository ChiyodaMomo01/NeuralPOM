"""Full endpoint caches and training-only calibration for P0-fixed campaigns."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field as dataclass_field
import json
import math
import os
import uuid
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from neuralpom.hybrid.constants import INPUT_PACK_ORDER, INPUT_THREED_FIELDS
from neuralpom.hybrid.state import StateNormalizer, pack_pom_batch
from neuralpom.training.daily_objective import adjacent_error_moments, state_metrics
from .checkpoint import atomic_json, atomic_torch, capture, restore
from .data import CORE, NUMERICS_VERSION, NOMINAL_CLOCK_TOL_DAYS, FixedCoreDataset
from .core_spec import core_spec


@dataclass(frozen=True)
class CacheContract:
    core: str
    numerics_version: str
    frequency_seconds: int
    grid: tuple[int, int, int]
    calibration_id: str
    data_identity: dict
    physics_settings: dict = dataclass_field(default_factory=dict)
    schema: str = 'p0-fixed-full-endpoint-v1'
    physical_step_seconds: int = 600
    input_semantics: str = 'physics-endpoint-before-correction'

    def __post_init__(self):
        if self.numerics_version != core_spec(self.core).numerics_version:
            raise ValueError('Cache requires the selected core numerical contract')
        if self.frequency_seconds not in (3600, 86400):
            raise ValueError('Cache frequency must be 1 h or 1 day')
        if len(self.grid) != 3 or min(self.grid) < 1:
            raise ValueError('Invalid cache grid')
        if not self.calibration_id or any(char not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for char in self.calibration_id):
            raise ValueError('Invalid calibration identity')
        if (self.data_identity.get('core') != self.core or
                self.data_identity.get('numerics_version') != self.numerics_version):
            raise ValueError('Cache data identity differs from its numerical core')

    @property
    def namespace(self):
        grid = 'x'.join(map(str, self.grid))
        return f'{self.core}-{self.frequency_seconds}s-{grid}-{self.calibration_id}'

    def metadata(self):
        return json.loads(json.dumps(asdict(self)))


def _canonical_physics_settings(settings):
    settings = dict(settings)
    acceleration = dict(settings['acceleration'])
    if acceleration.get('mode', 'eager') == 'eager':
        acceleration.pop('compiled_kernels', None)
    else:
        acceleration.setdefault('compiled_kernels', ['proft'])
    settings['acceleration'] = acceleration
    return settings


def physics_settings(config, core):
    """Readable numerical and arithmetic settings, including environment effects."""
    names = ('dti', 'dte', 'isplit', 'alpha', 'smoth', 'horcon',
             'spatial_shapiro_enabled', 'barotropic_shapiro', 'tracer_shapiro',
             'velocity_shapiro', 'eta_solver_rel_tol', 'eta_solver_preconditioner')
    settings = {name: getattr(core, name, None) for name in names}
    settings.update(autograd_version=getattr(core, 'AUTOGRAD_VERSION', 'torch-autograd-eager-v1'),
                    field_dtype=str(core.t.dtype),
                    acceleration=dict(config['acceleration']),
                    float32_matmul_precision=torch.get_float32_matmul_precision(),
                    cuda_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                    cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32))
    return _canonical_physics_settings(settings)


def asset_metadata(config, dataset, core=None):
    from .sampling import stage_samples
    samples = stage_samples(config, 'pretrain')
    spec = core_spec(config['core'])
    metadata = dict(sample_count=len(samples), samples=[list(sample) for sample in samples],
                core=spec.name, numerics_version=spec.numerics_version,
                frequency_seconds=config['interval_seconds'],
                grid=[config['grid']['im'], config['grid']['jm'], 16],
                data_identity=dataset.data_identity,
                state_interface=('p0-fixed-full-endpoint-v1' if spec.name == CORE else 'explicit-full-endpoint-v1'),
                input_semantics='physics-endpoint-before-correction',
                training_only=True, split=config['split'])
    if core is not None:
        metadata['physics_settings'] = physics_settings(config, core)
    return metadata


def validate_asset_metadata(metadata, config, dataset, core=None):
    expected = asset_metadata(config, dataset, core)
    for key, value in expected.items():
        observed = metadata.get(key)
        if key == 'physics_settings' and observed is not None:
            observed = _canonical_physics_settings(observed)
        if observed != value:
            raise ValueError(f'Calibration asset contract mismatch: {key}')
    if not isinstance(metadata.get('calibration_id'), str) or not metadata['calibration_id']:
        raise ValueError('Missing calibration identity')


def make_cache_contract(config, dataset, core=None):
    spec = core_spec(config['core'])
    identities, numerical_settings = [], []
    for key in ('scales', 'front_scales'):
        metadata = json.loads(Path(config[key]).read_text())['metadata']
        validate_asset_metadata(metadata, config, dataset, core)
        numerical_settings.append(_canonical_physics_settings(metadata['physics_settings']))
        identities.append(metadata['calibration_id'])
    with np.load(config['input_stats'], allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata_json'].item()))
        validate_asset_metadata(metadata, config, dataset, core)
        numerical_settings.append(_canonical_physics_settings(metadata['physics_settings']))
        identities.append(metadata['calibration_id'])
    if len(set(identities)) != 1 or any(value != numerical_settings[0] for value in numerical_settings):
        raise ValueError('Normalization assets belong to different calibrations or numerical settings')
    return CacheContract(spec.name, spec.numerics_version, config['interval_seconds'],
                         (config['grid']['im'], config['grid']['jm'], 16),
                         identities[0], dataset.data_identity, numerical_settings[0],
                         schema=('p0-fixed-full-endpoint-v1' if spec.name == CORE else 'explicit-full-endpoint-v1'))


class EndpointCache:
    def __init__(self, root, contract):
        self.contract = contract
        self.root = Path(root) / contract.namespace

    def path(self, member, model_day):
        return self.root / f'm{member}_d{model_day:04d}.pt'

    def _validate(self, member, model_day, payload):
        if payload.get('member') != member or payload.get('model_day') != model_day:
            raise ValueError('Cache sample identity mismatch')
        if payload.get('contract') != self.contract.metadata():
            raise ValueError('Cache contract mismatch')
        state = payload.get('state', {})
        if state.get('complete') is not True:
            raise ValueError('Endpoint cache must retain complete restart state')
        initial = float(payload.get('initial_time', math.nan))
        endpoint = float(state.get('scalars', {}).get('time', math.nan))
        if not math.isfinite(initial) or abs(initial - model_day) > NOMINAL_CLOCK_TOL_DAYS:
            raise ValueError('Cache initial clock mismatch')
        if not math.isfinite(endpoint) or abs(endpoint - initial - self.contract.frequency_seconds / 86400.) > NOMINAL_CLOCK_TOL_DAYS:
            raise ValueError('Cache endpoint clock mismatch')
        packed = payload.get('packed')
        if not torch.is_tensor(packed) or tuple(packed.shape) != (1, 101, *self.contract.grid[:2]):
            raise ValueError('Cache input tensor shape mismatch')
        if not bool(torch.isfinite(packed).all()):
            raise ValueError('Nonfinite cached input')
        if tuple(state.get('tensors', {}).get('u', torch.empty(0)).shape) != self.contract.grid:
            raise ValueError('Cache physical grid mismatch')

    def store(self, member, model_day, payload):
        payload = dict(payload, member=member, model_day=model_day, contract=self.contract.metadata())
        self._validate(member, model_day, payload)
        atomic_torch(self.path(member, model_day), payload)

    def load(self, member, model_day, map_location='cpu'):
        payload = torch.load(self.path(member, model_day), map_location=map_location, weights_only=False)
        self._validate(member, model_day, payload)
        return payload


class InputMoments:
    """Area/volume-weighted per-channel statistics of training endpoints."""
    def __init__(self, core):
        self.shapes = {field: tuple(getattr(core, field).shape) for field in INPUT_PACK_ORDER}
        self.moments = {field: torch.zeros((3, self.shapes[field][-1] if field in INPUT_THREED_FIELDS else 1),
                                          dtype=torch.float64, device=core.t.device) for field in INPUT_PACK_ORDER}

    def update(self, core):
        for field, moments in self.moments.items():
            value = getattr(core, field).detach().double()
            mask = core.dum if field in ('u', 'ub', 'ua', 'uab') else core.dvm if field in ('v', 'vb', 'va', 'vab') else core.fsm
            weights = (core.art * mask).detach().double()
            if field in INPUT_THREED_FIELDS:
                weights = (weights * (core.h + core.et).detach().double())[..., None].expand_as(value)
            else:
                value, weights = value[..., None], weights[..., None]
            if not bool(torch.isfinite(value).all()) or not bool((weights >= 0).all()):
                raise ValueError('Invalid state or geometry in input calibration')
            moments[0] += weights.sum((0, 1))
            moments[1] += (weights * value).sum((0, 1))
            moments[2] += (weights * value.square()).sum((0, 1))

    def arrays(self, floor=1e-6):
        result = {'format_version': np.array(2)}
        for field, moments in self.moments.items():
            if not bool((moments[0] > 0).all()):
                raise ValueError(f'No wet samples for input field {field}')
            mean = moments[1] / moments[0]
            std = (moments[2] / moments[0] - mean.square()).clamp_min(0).sqrt().clamp_min(floor)
            for name, value in (('mean', mean), ('scale', std)):
                shape = self.shapes[field]
                if field not in INPUT_THREED_FIELDS:
                    value = value[0]
                result[f'{name}__{field}'] = value.expand(shape).float().cpu().numpy().copy()
        return result


def _atomic_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def collective_call(function, device, distributed):
    """Propagate local I/O or logging failures before the next collective."""
    if not distributed:
        return function()
    result, failure = None, None
    try:
        result = function()
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
    failed = torch.tensor(int(failure is not None), device=device, dtype=torch.int32)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if int(failed):
        failures = [None] * dist.get_world_size()
        dist.all_gather_object(failures, failure)
        raise RuntimeError('Distributed cache preparation failed: ' +
                           '; '.join(f'rank {rank}: {error}' for rank, error in enumerate(failures) if error))
    return result


def build_assets(config, core, reset_callback, advance_callback, samples, validation_samples=(), logger=None):
    """Cache pure endpoints and calibrate on training samples only.

    Callbacks mutate ``core``. Ranks use unpadded shards and synchronize errors
    after each I/O phase, including rank-zero metadata writes and W&B logging.
    """
    from .sampling import stage_samples
    distributed = dist.is_available() and dist.is_initialized()
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    def phase(function):
        return collective_call(function, core.device, distributed)
    samples, validation_samples = list(samples), list(validation_samples)
    def preflight():
        data = FixedCoreDataset(config['data_root'], config['interval_seconds'], config['core'])
        if samples != stage_samples(config, 'pretrain'):
            raise ValueError('Calibration must use the complete registered pretrain sample list')
        if set(samples) & set(validation_samples) or len(set(validation_samples)) != len(validation_samples):
            raise ValueError('Duplicate or overlapping calibration/validation samples')
        split = config['split']
        for member, day in validation_samples:
            if (member not in split['validation_members'] or not split['validation_start_day'] <= day
                    or day + config['interval_seconds'] / 86400. >= split['validation_end_day']):
                raise ValueError('Validation sample is outside the held-out split')
        spec = core_spec(config['core'])
        if getattr(core, 'NUMERICS_VERSION', None) != spec.numerics_version or float(core.dti) != spec.physical_step_seconds:
            raise ValueError('Calibration core differs from registered production configuration')
        return data
    dataset = phase(preflight)
    metadata = asset_metadata(config, dataset, core)
    staging = Path(config['run_root']) / 'cache_staging' / config['frequency']
    calibration_manifest = staging / 'calibration.json'
    def register_calibration():
        if rank != 0:
            return
        if calibration_manifest.exists():
            validate_asset_metadata(json.loads(calibration_manifest.read_text()), config, dataset, core)
        else:
            atomic_json(calibration_manifest, dict(metadata, calibration_id=str(uuid.uuid4())))
    phase(register_calibration)
    def read_calibration():
        value = json.loads(calibration_manifest.read_text())
        validate_asset_metadata(value, config, dataset, core)
        return value
    metadata = phase(read_calibration)
    def prepare_moments():
        return (InputMoments(core),
                {field: torch.zeros(2, dtype=torch.float64, device=core.device)
                 for field in ('u', 'ub', 'v', 'vb', 't', 's', 'el', 'pressure')},
                torch.zeros((4, 2), dtype=torch.float64, device=core.device))
    moments, residual, fronts = phase(prepare_moments)
    all_samples = samples + validation_samples
    train_set = set(samples)
    def cache_endpoint(ordinal):
        if ordinal >= len(all_samples):
            return
        member, day = all_samples[ordinal]
        path = staging / f'm{member}_d{day:04d}.pt'
        if path.exists():
            item = torch.load(path, map_location='cpu', weights_only=False)
            if item.get('metadata') != metadata or item.get('sample') != (member, day):
                raise ValueError('Staging endpoint provenance mismatch')
            restore(core, item['state'])
        else:
            reset_callback(member, day)
            initial_time = float(core.time)
            if abs(initial_time - day) > NOMINAL_CLOCK_TOL_DAYS:
                raise ValueError('Calibration initial clock mismatch')
            with torch.no_grad():
                advance_callback(config['interval_seconds'])
            item = dict(state=capture(core), initial_time=initial_time, sample=(member, day), metadata=metadata)
            atomic_torch(path, item)
        if abs(float(core.time) - day - config['interval_seconds'] / 86400.) > NOMINAL_CLOCK_TOL_DAYS:
            raise ValueError('Calibration endpoint clock mismatch')
        if (member, day) in train_set:
            reference = dataset.load_target(member, day, config['interval_seconds'], map_location=core.device)
            moments.update(core)
            with torch.no_grad():
                for field, value in state_metrics(core, reference).items():
                    residual[field][0] += value.double()
                    residual[field][1] += 1
                fronts.add_(adjacent_error_moments(core, reference).double())
        if logger is not None and rank == 0:
            logger.log({'cache/endpoints_completed': min(len(all_samples), ordinal + world),
                        'cache/model_day': day, 'cache/member': member})
    slots = math.ceil(len(all_samples) / world)
    for slot in range(slots):
        phase(lambda: cache_endpoint(slot * world + rank))
    if distributed:
        for value in (*moments.moments.values(), *residual.values(), fronts):
            dist.all_reduce(value)
    def save_calibration():
        if rank != 0:
            return
        arrays = moments.arrays()
        arrays['metadata_json'] = np.array(json.dumps(metadata, sort_keys=True))
        _atomic_npz(config['input_stats'], arrays)
        scales = {field: float((value[0] / value[1]).sqrt().clamp_min(1e-8)) for field, value in residual.items()}
        if not all(math.isfinite(value) and value > 0 for value in scales.values()):
            raise ValueError('Nonfinite residual scales')
        atomic_json(config['scales'], dict(rms=scales, metadata=metadata))
        front_rms = (fronts[:, 0] / fronts[:, 1].clamp_min(1e-20)).sqrt().clamp_min(1e-8)
        atomic_json(config['front_scales'], dict(rms=front_rms.cpu().tolist(), metadata=metadata))
    phase(save_calibration)
    def load_calibration():
        contract = make_cache_contract(config, dataset, core)
        return (contract, EndpointCache(Path(config['run_root']) / 'cache', contract),
                StateNormalizer.from_path(config['input_stats'], core.device))
    contract, cache, normalizer = phase(load_calibration)
    def pack_endpoint(ordinal):
        if ordinal >= len(all_samples):
            return
        member, day = all_samples[ordinal]
        path = staging / f'm{member}_d{day:04d}.pt'
        item = torch.load(path, map_location='cpu', weights_only=False)
        restore(core, item['state'])
        packed = pack_pom_batch([core], normalizer).detach().cpu()
        cache.store(member, day, dict(state=item['state'], packed=packed, initial_time=item['initial_time'],
                    initial_path=str(dataset.initial_path(member, day)),
                    reference_path=str(dataset.label_path(member, day, config['interval_seconds']))))
    for slot in range(slots):
        phase(lambda: pack_endpoint(slot * world + rank))
    def complete():
        if rank == 0:
            atomic_json(cache.root / 'manifest.json', dict(contract=asdict(contract),
                        namespace=contract.namespace, train_samples=samples,
                        validation_samples=validation_samples, complete=True))
    phase(complete)
    return cache

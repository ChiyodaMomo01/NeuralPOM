"""One endpoint-correction state contract for preparation, training and inference."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from neuralpom.hybrid import build_initialized_core, inject_core_state, StateNormalizer, pack_pom_batch
from neuralpom.hybrid.sparse_coupling import apply_joint_correction, volume_weights, LEVELS
from neuralpom.hybrid.sparse_physics import SparsePhysics
from .checkpoint import capture, restore
from .data import FixedCoreDataset, NOMINAL_CLOCK_TOL_DAYS
from .core_spec import core_spec
from .losses import build_loss
from .telemetry import CorrectionAccumulator


class Runtime:
    def __init__(self, config, device, rank=0, assets=True):
        self.config, self.device = config, torch.device(device)
        self.interval_seconds = config['interval_seconds']
        self.physical_forward_steps = 0
        self.spec = core_spec(config['core'])
        grid = config['grid']
        self.core = build_initialized_core(self.spec.core_class, self.device,
                                           grid['im'], grid['jm'])
        self.spec.configure(self.core)
        self.baseline = capture(self.core, cpu=False)
        accel = config['acceleration']
        self.physics = SparsePhysics(self.device, rank=rank, cuda_graph=accel['cuda_graph'],
            chunk_steps=accel['chunk_steps'], acceleration=accel['mode'],
            checkpoint_physics=accel['checkpoint_physics'], cg_backend=accel.get('cg_backend', 'torch'),
            compiled_kernels=tuple(accel.get('compiled_kernels', ['proft'])), core_name=self.spec.name)
        self.data = FixedCoreDataset(config['data_root'], self.interval_seconds, self.spec.name)
        if self.data.grid != (grid['im'], grid['jm'], self.core.kb):
            raise ValueError('Dataset and physical core grids differ')
        self.normalizer = None
        if assets:
            self.load_assets()

    def load_assets(self):
        from .cache import EndpointCache, make_cache_contract
        self.cache_contract = make_cache_contract(self.config, self.data, self.core)
        self.cache = EndpointCache(Path(self.config.get('cache_root', Path(self.config['run_root']) / 'cache')), self.cache_contract)
        self.normalizer = StateNormalizer.from_path(self.config['input_stats'], self.device)
        value = json.loads(Path(self.config['scales']).read_text())
        rms = value['rms']
        self.scales = rms[str(self.interval_seconds // 3600)] if str(self.interval_seconds // 3600) in rms else rms
        front = json.loads(Path(self.config['front_scales']).read_text())['rms']
        self.objective = build_loss(self.config, self.scales, front)
        self.correction_scales = {field: self.scales[field] for field in ('u', 'v', 't', 's')}

    def contract(self):
        return {'version': 'neuralpom-v1-endpoint-fullstate', 'core': self.spec.name,
                'numerics_version': self.spec.numerics_version, 'interval_seconds': self.interval_seconds,
                'autograd_version': getattr(self.core, 'AUTOGRAD_VERSION', 'torch-autograd-eager-v1'),
                'physical_step_seconds': self.spec.physical_step_seconds, 'grid': self.config['grid'],
                'input_channels': 101, 'output_channels': 60, 'levels': 15,
                'hidden_dim': self.config['model']['hidden_dim'],
                'calibration_id': self.normalizer.metadata['calibration_id'],
                'physics_settings': self.cache_contract.physics_settings,
                'data_identity': self.data.data_identity,
                'split': self.config['split'],
                'correction': 'endpoint-paired-history-eos-zero-column-u-v-layer-budget-ts-cap5'}

    def reset(self, member, model_day):
        restore(self.core, self.baseline)
        initial = self.data.load_initial(member, model_day, map_location=self.device)
        inject_core_state(self.core, initial)
        self.core.dt = self.core.h + self.core.et
        self.core.d = self.core.h + self.core.el
        self.assert_time(float(model_day))
        return float(self.core.time)

    def load_cached(self, member, model_day):
        record = self.cache.load(member, model_day)
        restore(self.core, record['state'])
        self.assert_time(model_day + self.interval_seconds / 86400)
        return record

    def require_cache(self, samples, validation_samples=()):
        manifest = json.loads((self.cache.root / 'manifest.json').read_text())
        if not manifest.get('complete') or manifest.get('namespace') != self.cache_contract.namespace:
            raise ValueError('Training requires a complete endpoint cache for this calibration')
        registered = {tuple(item) for item in manifest['train_samples']}
        held_out = {tuple(item) for item in manifest['validation_samples']}
        if not set(samples) <= registered or not set(validation_samples) <= held_out:
            raise ValueError('The endpoint cache does not cover this stage sample list')

    def advance(self, differentiable=False, seconds=None):
        seconds = self.interval_seconds if seconds is None else seconds
        steps, rest = divmod(seconds, self.spec.physical_step_seconds)
        if rest or steps <= 0:
            raise ValueError('Advance must use positive whole physical steps')
        start = float(self.core.time)
        self.physics.advance(self.core, steps=int(steps), differentiable=differentiable)
        self.assert_time(start + seconds / 86400)
        self.physical_forward_steps += int(steps)

    def observation(self):
        return pack_pom_batch([self.core], self.normalizer).to(self.device)

    def correct(self, model, packed=None):
        packed = self.observation() if packed is None else packed.to(self.device)
        return self.apply_raw(model(packed)[0])

    def apply_raw(self, raw, *, density75=False):
        """Decode the same raw action for both posttrain methods."""
        before = {field: getattr(self.core, field) for field in ('u', 'v', 't', 's')}
        if density75:
            from neuralpom.evaluation.density import apply_density75
            diagnostics = apply_density75(self.core, raw, self.correction_scales)
        else:
            diagnostics = apply_joint_correction(self.core, raw, self.correction_scales)
        with torch.no_grad():
            for field, original in before.items():
                delta = (getattr(self.core, field) - original)[..., :LEVELS]
                weights = volume_weights(self.core, field)
                diagnostics[field + '_correction_rms'] = ((delta.square() * weights).sum() / weights.sum()).sqrt()
                diagnostics[field + '_correction_max_abs'] = delta.abs().max()
                if field in ('t', 's'):
                    diagnostics[field + '_actual_inventory_delta'] = (delta * weights).sum()
                    diagnostics[field + '_actual_mean_delta'] = (delta * weights).sum() / weights.sum()
        return diagnostics

    @torch.no_grad()
    def training_diagnostics(self, member, model_day):
        from .diagnostics import FIELDS, state_diagnostics
        lead = round((float(self.core.time) - model_day) * 86400)
        reference = self.data.load_target(member, model_day, lead, map_location=self.device)
        self.assert_time(float(reference['time']))
        prediction = {field: getattr(self.core, field).detach() for field in FIELDS}
        metrics, profiles, spectra = state_diagnostics(self.core, prediction, reference)
        return {'member': member, 'start_day': model_day, 'lead_seconds': lead,
                'time': float(self.core.time), 'reference_time': float(reference['time']),
                'metrics': metrics, 'profiles': profiles, 'velocity_spectra': spectra}

    def score(self, member, model_day, lead_seconds, include_eta=True):
        reference = self.data.load_target(member, model_day, lead_seconds, map_location=self.device)
        self.assert_time(float(reference['time']))
        return self.objective(self.core, reference, include_eta=include_eta)

    def assert_time(self, expected):
        if abs(float(self.core.time) - expected) > NOMINAL_CLOCK_TOL_DAYS:
            raise ValueError(f'Physical clock {self.core.time} differs from expected {expected}')

    def detach(self):
        self.physics.release_completed_rollout(self.core)

    def snapshot(self):
        return capture(self.core)

    def restore(self, state):
        restore(self.core, state)


@torch.no_grad()
def frozen_rollout(runtime, model, member, model_day, days, baseline=False, on_intraday=None):
    """Yield corrected fields and scores; a physics baseline skips correction."""
    was_training = model.training
    model.eval()
    runtime.reset(member, model_day)
    interval = runtime.interval_seconds
    steps, rest = divmod(int(days * 86400), interval)
    if rest or steps <= 0:
        raise ValueError('Inference duration must contain whole correction intervals')
    fields = ('u', 'v', 't', 's', 'el')
    budgets = CorrectionAccumulator()
    try:
        for index in range(1, steps + 1):
            runtime.advance(False)
            lead = index * interval
            daily = lead % 86400 == 0
            intraday = on_intraday is not None and interval == 3600 and 0 < lead % 86400 <= 14400
            diagnostics = {}
            if not baseline:
                diagnostics = runtime.correct(model)
                budgets.add(diagnostics)
            if daily or intraday:
                _, post = runtime.score(member, model_day, lead)
                after = {key: getattr(runtime.core, key).detach().cpu().clone() for key in fields}
                metrics = {'post/' + key: float(value) for key, value in post.items()}
            if intraday:
                reference = runtime.data.load_target(member, model_day, lead, map_location='cpu')
                on_intraday({'lead_seconds': lead, 'time': float(runtime.core.time),
                    'reference_time': float(reference['time']),
                    'within_day_hour': (lead % 86400) // 3600,
                    'post': after,
                    'reference': {key: reference[key] for key in fields},
                    'correction_diagnostics': {key: float(value) for key, value in diagnostics.items()},
                    'metrics': metrics})
            if daily:
                yield {'lead_day': lead // 86400, 'time': float(runtime.core.time),
                       'post': after, 'correction_diagnostics': budgets.summary(),
                       'metrics': metrics}
                budgets.clear()
    finally:
        model.train(was_training)

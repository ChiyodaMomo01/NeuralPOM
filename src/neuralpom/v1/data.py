"""P0-fixed reference data indexed by model time, never by file-day aliases."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch

from .core_spec import core_spec

CORE = 'implicit_p0_fixed'
NUMERICS_VERSION = 'implicit-dg-v2-aligned-1e5-p0-barotropic-coupling-fixed'
# Accumulated fine-step clock drift reaches ~1e-7 day near production day 9000.
NOMINAL_CLOCK_TOL_DAYS = 2e-7
LABEL_FIELDS = ('u', 'ub', 'v', 'vb', 't', 's', 'el', 'ua', 'uab', 'va', 'vab')
INITIAL_3D_FIELDS = ('u', 'ub', 'v', 'vb', 't', 'tb', 's', 'sb', 'q2', 'q2b', 'q2l', 'q2lb', 'km', 'kh', 'kq', 'aam', 'rho', 'w')
INITIAL_2D_FIELDS = ('el', 'elb', 'et', 'etb', 'etf', 'ua', 'uab', 'va', 'vab', 'utf', 'utb', 'vtf', 'vtb', 'adx2d', 'ady2d', 'drx2d', 'dry2d', 'vfluxf', 'vfluxb', 'wubot', 'wvbot', 'egb', 'uaf', 'vaf', 'elf')



class FixedCoreDataset:
    def __init__(self, root, frequency_seconds=86400, core_name=CORE):
        self.root = Path(root)
        self.spec = core_spec(core_name)
        self.frequency_seconds = int(frequency_seconds)
        if self.frequency_seconds not in (3600, 86400):
            raise ValueError('Correction interval must be 1 h or 1 day')
        manifest_path = self.root / 'manifest.json'
        self.manifest = json.loads(manifest_path.read_text())
        if (self.manifest.get('generation_core_requested') != self.spec.name or
                self.manifest.get('generation_core_numerics_version') != self.spec.numerics_version):
            raise ValueError('Dataset does not match the registered dynamical core')
        if (self.spec.name == CORE and
                str(self.manifest.get('implicit_spatial_shapiro_requested', '')).lower() not in ('0', 'false', 'off', 'no')):
            raise ValueError('V1 requires data generated with spatial Shapiro OFF')
        if self.spec.name == 'explicit':
            integration = self.manifest.get('integration', {})
            if (integration.get('dte_seconds') != 5. or integration.get('isplit') != 30 or
                    integration.get('dti_seconds') != 150.):
                raise ValueError('Explicit source trajectory has the wrong time split')
        self.data_identity = dict(root=str(self.root.resolve()),
                                  core=self.manifest['generation_core_requested'],
                                  numerics_version=self.manifest['generation_core_numerics_version'],
                                  generation_variant=self.manifest.get('generation_variant'),
                                  downsampling_algorithm=self.manifest.get('downsampling', {}).get('algorithm'))
        self.grid = tuple(self.manifest.get('downsampling', {}).get('coarse_grid', (90, 82))) + (16,)
        self._verified_members = set()

    def _member(self, member):
        if not isinstance(member, int) or not 0 <= member < 8:
            raise ValueError('Member must be an integer in [0, 7]')
        base = self.root / f'member_{member}_gpu{member}'
        if member not in self._verified_members:
            provenance = json.loads((base / 'provenance.json').read_text())
            if (provenance.get('member') != member or provenance.get('core_requested') != self.spec.name or
                    provenance.get('core_numerics_version') != self.spec.numerics_version):
                raise ValueError(f'Member core provenance mismatch: {base}')
            numerics = provenance.get('resolved_numerics', {})
            if self.spec.name == CORE and (numerics.get('spatial_shapiro_enabled') is not False or
                    any(numerics.get(key) != 0. for key in ('barotropic_shapiro_gamma',
                        'tracer_shapiro_gamma', 'velocity_shapiro_gamma'))):
                raise ValueError(f'Member data requires spatial Shapiro OFF: {base}')
            if self.spec.name == 'explicit' and any(numerics.get(key) != value for key, value in
                    (('alpha', self.spec.alpha), ('smoth', self.spec.smoth), ('horcon', self.spec.horcon),
                     ('dte_seconds', 5.), ('dti_seconds', 150.), ('isplit', 30))):
                raise ValueError(f'Explicit member numerical settings mismatch: {base}')
            algorithm = self.data_identity['downsampling_algorithm']
            if algorithm is not None and provenance.get('downsampling_algorithm') != algorithm:
                raise ValueError(f'Member downsampling differs from dataset manifest: {base}')
            self._verified_members.add(member)
        return base

    def initial_path(self, member, model_day):
        if int(model_day) != model_day or not 1 <= model_day <= 9000:
            raise ValueError('Initial model day must be an integer in [1, 9000]')
        return self._member(member) / 'checkpoints_lowres_consistent' / f'checkpoint_Day{int(model_day)-1:04d}.npz'

    def label_path(self, member, model_day, lead_seconds):
        if int(model_day) != model_day or lead_seconds < 0 or int(lead_seconds) != lead_seconds:
            raise ValueError('Labels require integer model days and nonnegative integer lead seconds')
        whole_days, within_day = divmod(int(lead_seconds), 86400)
        day = int(model_day) + whole_days
        if within_day == 0:
            return self.initial_path(member, day)
        if within_day > 14400 or within_day % 1200:
            raise ValueError('Intraday labels exist only every 20 min in the first 4 h')
        if not 0 <= day <= 8999:
            raise ValueError('Reference day outside registered coverage')
        segment = within_day // 1200 - 1
        return self._member(member) / 'reference_data_lowres_consistent' / f'ref_Day{day:04d}_seg{segment:02d}.npz'

    @staticmethod
    def _read(path, expected_time, map_location, fields=None):
        with np.load(path, allow_pickle=False) as archive:
            if 'time' not in archive or abs(float(archive['time']) - expected_time) > NOMINAL_CLOCK_TOL_DAYS:
                raise ValueError(f'Reference clock mismatch: {path}; expected {expected_time}')
            keys = archive.files if fields is None else (*fields, 'time')
            result = {key: torch.as_tensor(archive[key].copy(), device=map_location,
                      dtype=torch.float64 if key == 'time' else torch.float32) for key in keys}
        if not all(bool(torch.isfinite(value).all()) for value in result.values()):
            raise ValueError(f'Nonfinite reference: {path}')
        return result

    def _validate_shapes(self, state, *, initial):
        fields_3d = INITIAL_3D_FIELDS if initial else LABEL_FIELDS[:6]
        fields_2d = INITIAL_2D_FIELDS if initial else LABEL_FIELDS[6:]
        scalars = ('time', 'iint') if initial else ('time',)
        required = (*fields_3d, *fields_2d, *scalars)
        missing = set(required) - set(state)
        if missing:
            raise ValueError(f'Missing required physical fields: {sorted(missing)}')
        for field in required:
            expected = self.grid if field in fields_3d else self.grid[:2] if field in fields_2d else ()
            if tuple(state[field].shape) != expected:
                raise ValueError(f'Physical field {field} shape {tuple(state[field].shape)} != {expected}')
        return state

    def load_initial(self, member, model_day, map_location='cpu'):
        state = self._read(self.initial_path(member, model_day), float(model_day), map_location)
        return self._validate_shapes(state, initial=True)

    def load_target(self, member, model_day, lead_seconds, map_location='cpu'):
        state = self._read(self.label_path(member, model_day, lead_seconds),
                           model_day + lead_seconds / 86400., map_location, LABEL_FIELDS)
        return self._validate_shapes(state, initial=False)

    target = load_target

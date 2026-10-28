"""Data-time, normalization, cache isolation, and differentiable loss contracts."""
import json
from datetime import timedelta
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from neuralpom.v1.data import CORE, NUMERICS_VERSION, FixedCoreDataset, INITIAL_3D_FIELDS, INITIAL_2D_FIELDS
from neuralpom.v1.cache import CacheContract, EndpointCache, InputMoments, validate_asset_metadata, asset_metadata, build_assets, collective_call, make_cache_contract
from neuralpom.v1.config import resolve_config, training_settings
from neuralpom.v1.sampling import stage_samples
from neuralpom.v1.losses import build_loss, spectral_loss, physical_loss
from neuralpom.training.daily_objective import daily_objective


def reference_root(root, member=0, core=CORE):
    (root / 'manifest.json').write_text(json.dumps({'generation_core_requested': core,
        'generation_core_numerics_version': NUMERICS_VERSION,
        'implicit_spatial_shapiro_requested': 'off',
        'downsampling': {'coarse_grid': [8, 6]}}))
    base = root / f'member_{member}_gpu{member}'
    base.mkdir(exist_ok=True)
    (base / 'provenance.json').write_text(json.dumps({'member': member, 'core_requested': core,
        'core_numerics_version': NUMERICS_VERSION,
        'resolved_numerics': {'spatial_shapiro_enabled': False, 'barotropic_shapiro_gamma': 0.,
                             'tracer_shapiro_gamma': 0., 'velocity_shapiro_gamma': 0.}}))
    return base


def small_core():
    shape = (8, 6, 16)
    core = SimpleNamespace(dz=torch.ones(16, dtype=torch.float64) / 15,
        art=torch.ones(shape[:2], dtype=torch.float64), h=torch.ones(shape[:2], dtype=torch.float64),
        et=torch.zeros(shape[:2], dtype=torch.float64),
        fsm=torch.ones(shape[:2], dtype=torch.float64), dum=torch.ones(shape[:2], dtype=torch.float64),
        dvm=torch.ones(shape[:2], dtype=torch.float64))
    for field in ('u', 'ub', 'v', 'vb', 't', 's'):
        setattr(core, field, torch.zeros(shape, dtype=torch.float64))
    for field in ('el', 'ua', 'uab', 'va', 'vab'):
        setattr(core, field, torch.zeros(shape[:2], dtype=torch.float64))
    return core


def failing_cache_rank(rank, init_file, output):
    dist.init_process_group('gloo', init_method='file://' + init_file, rank=rank, world_size=2,
                            timeout=timedelta(seconds=15))
    try:
        def operation():
            if rank == 0:
                raise OSError('injected rank-zero metadata failure')
        try:
            collective_call(operation, torch.device('cpu'), True)
        except RuntimeError as exc:
            Path(output, f'rank{rank}.txt').write_text(str(exc))
    finally:
        dist.destroy_process_group()


class DataContractTest(unittest.TestCase):
    def test_model_day_and_intraday_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_root(root)
            data = FixedCoreDataset(root, 3600)
            self.assertEqual(data.initial_path(0, 1000).name, 'checkpoint_Day0999.npz')
            for hour, segment in ((1, 2), (2, 5), (3, 8), (4, 11)):
                self.assertEqual(data.label_path(0, 1000, hour * 3600).name, f'ref_Day1000_seg{segment:02d}.npz')
            self.assertEqual(data.label_path(0, 1000, 86400).name, 'checkpoint_Day1000.npz')
            self.assertEqual(data.label_path(0, 1000, 25 * 3600).name, 'ref_Day1001_seg02.npz')
            for seconds in (0.5, -1, 5 * 3600, 23 * 3600):
                with self.assertRaises(ValueError):
                    data.label_path(0, 1000, seconds)

    def test_legacy_data_and_member_swap_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_root(root, core='implicit')
            with self.assertRaises(ValueError):
                FixedCoreDataset(root)
            base = reference_root(root)
            (base / 'provenance.json').write_text(json.dumps({'member': 1, 'core_requested': CORE,
                'core_numerics_version': NUMERICS_VERSION}))
            with self.assertRaises(ValueError):
                FixedCoreDataset(root).initial_path(0, 1000)

    def test_fixed_core_name_does_not_accept_shapiro_on_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = reference_root(root)
            manifest = json.loads((root / 'manifest.json').read_text())
            manifest['implicit_spatial_shapiro_requested'] = 'on'
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'Shapiro OFF'):
                FixedCoreDataset(root)
            reference_root(root)
            provenance = json.loads((base / 'provenance.json').read_text())
            provenance['resolved_numerics']['velocity_shapiro_gamma'] = .05
            (base / 'provenance.json').write_text(json.dumps(provenance))
            with self.assertRaisesRegex(ValueError, 'Shapiro OFF'):
                FixedCoreDataset(root).initial_path(0, 1000)

    def test_file_clock_is_verified_not_inferred(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = reference_root(root)
            checkpoints = base / 'checkpoints_lowres_consistent'
            checkpoints.mkdir()
            path = checkpoints / 'checkpoint_Day0999.npz'
            np.savez(path, time=np.array(999.), u=np.zeros((2, 2, 16)))
            data = FixedCoreDataset(root)
            with self.assertRaisesRegex(ValueError, 'clock'):
                data.load_initial(0, 1000)
            fields = {field: np.zeros((8, 6, 16)) for field in INITIAL_3D_FIELDS}
            fields.update({field: np.zeros((8, 6)) for field in INITIAL_2D_FIELDS})
            np.savez(path, time=np.array(1000. - 9e-8), iint=np.array(1), **fields)
            self.assertAlmostEqual(float(data.load_initial(0, 1000)['time']), 1000. - 9e-8)
            fields.pop('tb')
            np.savez(path, time=np.array(1000.), iint=np.array(1), **fields)
            with self.assertRaisesRegex(ValueError, 'required'):
                data.load_initial(0, 1000)
            fields['tb'] = np.zeros((8, 6, 15))
            np.savez(path, time=np.array(1000.), iint=np.array(1), **fields)
            with self.assertRaisesRegex(ValueError, 'shape'):
                data.load_initial(0, 1000)


class CacheContractTest(unittest.TestCase):
    def contract(self):
        return CacheContract(CORE, NUMERICS_VERSION, 3600, (2, 3, 16), 'test-calibration',
                             {'core': CORE, 'numerics_version': NUMERICS_VERSION})

    def payload(self):
        return {'initial_time': 1000., 'state': {'complete': True,
            'scalars': {'time': 1000 + 1 / 24}, 'objects': {},
            'tensors': {'u': torch.zeros(2, 3, 16), 'solver_warm_start': torch.ones(2, 3)}},
            'packed': torch.zeros(1, 101, 2, 3)}

    def test_complete_state_roundtrip_and_frequency_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = EndpointCache(directory, self.contract())
            cache.store(0, 1000, self.payload())
            value = cache.load(0, 1000)
            self.assertTrue(torch.equal(value['state']['tensors']['solver_warm_start'], torch.ones(2, 3)))
            daily = EndpointCache(directory, replace(self.contract(), frequency_seconds=86400))
            self.assertNotEqual(cache.root, daily.root)
            with self.assertRaises(FileNotFoundError):
                daily.load(0, 1000)
            value['contract'] = daily.contract.metadata()
            torch.save(value, cache.path(0, 1000))
            with self.assertRaisesRegex(ValueError, 'contract'):
                cache.load(0, 1000)

    def test_partial_and_wrong_clock_states_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = EndpointCache(directory, self.contract())
            value = self.payload()
            value['state']['complete'] = False
            with self.assertRaisesRegex(ValueError, 'complete'):
                cache.store(0, 1000, value)
            value = self.payload()
            value['state']['scalars']['time'] = 1001
            with self.assertRaisesRegex(ValueError, 'clock'):
                cache.store(0, 1000, value)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo unavailable')
    def test_rank_zero_io_failure_reaches_all_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            context = mp.spawn(failing_cache_rank, args=(str(Path(directory) / 'rendezvous'), directory),
                               nprocs=2, join=False)
            # The collective error exchange must finish; a waiting peer is a failure.
            import time
            deadline = time.monotonic() + 25
            while time.monotonic() < deadline and not context.join(timeout=1):
                pass
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
                    self.fail('Cache error left a distributed peer waiting')
            for rank in range(2):
                self.assertIn('rank-zero metadata failure', (Path(directory) / f'rank{rank}.txt').read_text())

    def test_channel_moments_use_wet_volume_and_area(self):
        core = small_core()
        core.fsm[:] = 0
        core.dum[:] = 0
        core.dvm[:] = 0
        for mask in (core.fsm, core.dum, core.dvm):
            mask[0, :2] = 1
        core.h[0, 1] = 3
        for field in ('u', 'ub', 'v', 'vb', 't', 's'):
            getattr(core, field)[:] = 1000  # dry values must not enter statistics
            getattr(core, field)[0, 0] = 2
            getattr(core, field)[0, 1] = 6
        for field in ('el', 'ua', 'uab', 'va', 'vab'):
            getattr(core, field)[:] = 1000
            getattr(core, field)[0, 0] = 2
            getattr(core, field)[0, 1] = 6
        moments = InputMoments(core)
        moments.update(core)
        arrays = moments.arrays()
        self.assertAlmostEqual(float(arrays['mean__t'][0, 0, 0]), 5.)
        self.assertAlmostEqual(float(arrays['scale__t'][0, 0, 0]), np.sqrt(3), places=6)
        self.assertAlmostEqual(float(arrays['mean__el'][0, 0]), 4.)
        self.assertAlmostEqual(float(arrays['scale__el'][0, 0]), 2.)


    def test_asset_builder_training_only_and_cached_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / 'references'
            data_root.mkdir()
            reference_root(data_root, 0)
            reference_root(data_root, 7)
            config = resolve_config({'data_root': str(data_root), 'run_root': str(root / 'run'),
                'grid': {'im': 8, 'jm': 6}, 'world_size': 1,
                'split': {'train_members': [0], 'train_start_day': 1000, 'train_end_day': 1008}})
            core = small_core()
            core.device = torch.device('cpu')
            core.NUMERICS_VERSION = NUMERICS_VERSION
            core.dti = 600.
            core.time = 1000.
            samples = stage_samples(config, 'pretrain')
            self.assertEqual(samples, [(0, 1000), (0, 1006)])
            for member, day in samples:
                base = data_root / f'member_{member}_gpu{member}' / 'checkpoints_lowres_consistent'
                base.mkdir(exist_ok=True)
                fields = {field: np.zeros_like(getattr(core, field)) for field in
                          ('u', 'ub', 'v', 'vb', 't', 's', 'el', 'ua', 'uab', 'va', 'vab')}
                np.savez(base / f'checkpoint_Day{day-1:04d}.npz', time=np.array(float(day)), **fields)
                np.savez(base / f'checkpoint_Day{day:04d}.npz', time=np.array(float(day+1)), **fields)
            validation = [(7, 7000)]
            base = data_root / 'member_7_gpu7' / 'checkpoints_lowres_consistent'
            base.mkdir()
            np.savez(base / 'checkpoint_Day6999.npz', time=np.array(7000.), **fields)
            np.savez(base / 'checkpoint_Day7000.npz', time=np.array(7001.), **fields)
            calls = []
            def reset(member, day):
                core.time = float(day)
                value = 100. if member == 7 else 1. + (day - 1000) / 3
                for field in ('u', 'ub', 'v', 'vb', 't', 's', 'el', 'ua', 'uab', 'va', 'vab'):
                    getattr(core, field).fill_(value)
            def advance(seconds):
                calls.append(seconds)
                core.time += seconds / 86400.
            def metrics(core, reference):
                return {field: core.t.square().mean() for field in
                        ('u', 'ub', 'v', 'vb', 't', 's', 'el', 'pressure')}
            with patch('neuralpom.v1.cache.state_metrics', side_effect=metrics):
                cache = build_assets(config, core, reset, advance, samples, validation)
                self.assertEqual(len(calls), 3)
                build_assets(config, core, reset, advance, samples, validation)
                self.assertEqual(len(calls), 3)  # no repeated physical integration
            with np.load(config['input_stats'], allow_pickle=False) as archive:
                self.assertEqual(float(archive['mean__t'][0, 0, 0]), 2.)
                self.assertEqual(float(archive['scale__t'][0, 0, 0]), 1.)
            scales = json.loads(Path(config['scales']).read_text())
            self.assertAlmostEqual(scales['rms']['t'], np.sqrt(5))
            self.assertEqual(scales['metadata']['sample_count'], 2)
            self.assertTrue(scales['metadata']['training_only'])
            heldout = cache.load(7, 7000)
            self.assertEqual(float(heldout['packed'][0, 64, 0, 0]), 98.)
            # A different frequency or train split must not reuse these assets.
            data = FixedCoreDataset(data_root)
            changed = dict(config, interval_seconds=3600)
            with self.assertRaises(ValueError):
                validate_asset_metadata(scales['metadata'], changed, data)
            equivalent_eager = dict(config, acceleration=dict(config['acceleration'], compiled_kernels=['proft']))
            self.assertEqual(make_cache_contract(equivalent_eager, data, core).metadata(), cache.contract.metadata())
            core.eta_solver_rel_tol = 1e-3
            with self.assertRaisesRegex(ValueError, 'physics_settings'):
                make_cache_contract(config, data, core)
            del core.eta_solver_rel_tol
            changed_backend = dict(config, acceleration=dict(config['acceleration'], mode='compiled_vertical'))
            with self.assertRaisesRegex(ValueError, 'physics_settings'):
                make_cache_contract(changed_backend, data, core)


class LossTest(unittest.TestCase):
    def test_physical_mean_gradient_and_dry_mask(self):
        core = small_core()
        core.fsm[0, 0] = 0
        core.t = torch.ones_like(core.t, requires_grad=True)
        reference = {'t': torch.zeros_like(core.t), 's': torch.zeros_like(core.s), 'el': torch.zeros_like(core.el)}
        value = physical_loss(core, reference, {'t': 2., 's': 1., 'el': 1.})
        self.assertAlmostEqual(float(value), 1 / 12)
        value.backward()
        self.assertTrue(torch.isfinite(core.t.grad).all())
        self.assertEqual(float(core.t.grad[0, 0].abs().sum()), 0.)
        self.assertGreater(float(core.t.grad[1, 1, 0]), 0.)

    def test_spectrum_identity_and_zero_energy_gradient(self):
        core = small_core()
        core.u.requires_grad_(True)
        core.v.requires_grad_(True)
        reference = {'u': torch.zeros_like(core.u), 'v': torch.zeros_like(core.v)}
        loss = spectral_loss(core, reference)
        self.assertEqual(float(loss), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(core.u.grad).all())
        signal = torch.sin(torch.arange(8, dtype=torch.float64) * 2 * torch.pi / 8)[:, None, None]
        core.u = signal.expand_as(core.u).clone().requires_grad_()
        reference['u'] = .5 * core.u.detach()
        loss = spectral_loss(core, reference)
        self.assertGreater(float(loss), 0.)
        loss.backward()
        self.assertGreater(float(core.u.grad.abs().sum()), 0.)
        self.assertTrue(torch.isfinite(core.u.grad).all())

    def test_configured_weights_and_normalization(self):
        core = small_core()
        core.t = torch.ones_like(core.t, requires_grad=True)
        reference = {field: torch.zeros_like(getattr(core, field)) for field in ('u', 'v', 't', 's', 'el')}
        objective = build_loss({'losses': [{'name': 'physical', 'weight': 2., 'scale': 4.}]},
                               {'t': 1., 's': 1., 'el': 1.})
        value, details = objective(core, reference)
        self.assertAlmostEqual(float(value), 2 / 16 / 3)
        self.assertAlmostEqual(float(details['loss/physical']), 1 / 3)
        value.backward()
        self.assertGreater(float(core.t.grad.abs().sum()), 0.)
        for term in ({'name': 'other'}, {'name': 'physical', 'scale': 0}, {'name': 'physical', 'weight': -1}):
            with self.assertRaises(ValueError):
                build_loss({'losses': [term]}, {})

    def test_existing_daily_weighted_mse_and_gradient_preserved(self):
        core = small_core()
        core.t = torch.ones_like(core.t, requires_grad=True)
        reference = {field: torch.zeros_like(getattr(core, field)) for field in ('t', 's')}
        scales = {field: 1. for field in ('u', 'ub', 'v', 'vb', 't', 's', 'el', 'pressure')}
        # Isolate the composition contract from the EOS/pressure kernel, which
        # has its own physical-core regression tests.
        metrics = {field: (core.t.square().mean() if field == 't' else core.t.sum() * 0) for field in scales}
        with patch('neuralpom.training.daily_objective.state_metrics', return_value=metrics):
            expected, _ = daily_objective(core, reference, scales, torch.ones(4), include_eta=False)
            actual, _ = build_loss({'losses': [{'name': 'weighted_mse'}]}, scales, [1.] * 4)(core, reference, include_eta=False)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(torch.autograd.grad(actual, core.t, retain_graph=True)[0],
                                    torch.autograd.grad(expected, core.t)[0]))

    def test_actual_loss_config_is_part_of_restart_contract(self):
        base = {'data_root': '/tmp/fixed-data', 'run_root': '/tmp/fixed-run'}
        for old in ({'components': {'spectral': 1.}}, {}):
            config = dict(base, loss=old)
            with self.assertRaisesRegex(ValueError, 'legacy loss key'):
                build_loss(config, {})
            with self.assertRaises(ValueError):
                resolve_config(config)
        first = resolve_config(dict(base, losses=[{'name': 'weighted_mse'}]))
        second = resolve_config(dict(base, losses=[{'name': 'spectral'}]))
        self.assertNotEqual(training_settings(first), training_settings(second))
        self.assertEqual(build_loss(first, {}).terms[0][0], 'weighted_mse')
        self.assertEqual(build_loss(second, {}).terms[0][0], 'spectral')

    def test_duplicate_components_cannot_hide_contributions(self):
        for name in ('weighted_mse', 'spectral', 'physical'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Duplicate loss'):
                build_loss({'losses': [{'name': name, 'weight': 1.},
                                       {'name': name, 'weight': 2.}]}, {})


if __name__ == '__main__':
    unittest.main()

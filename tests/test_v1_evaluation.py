"""Scientific closure and paired evaluation tests, using the actual core EOS."""
from __future__ import annotations
import json
from datetime import timedelta
from unittest.mock import patch
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from neuralpom.cores.implicit_p0_fixed.momentum import MomentumMixin
from neuralpom.v1.diagnostics import FIELDS, state_diagnostics, velocity_spectra
from neuralpom.v1.evaluation import compare_to_physics, summarize_cases, evaluate
from neuralpom.v1.visualization import render_evaluation, render_comparison


class DiagnosticCore(MomentumMixin):
    def __init__(self):
        nx, ny = 8, 6
        self.kbm1, self.kb = 15, 16
        self.rhoref, self.grav, self.tbias, self.sbias = 1035., 9.81, 10., 35.
        self.h = torch.arange(nx * ny, dtype=torch.float32).reshape(nx, ny) + 100
        self.art = torch.arange(nx * ny, dtype=torch.float32).reshape(nx, ny) + 1
        self.fsm = torch.ones(nx, ny)
        self.fsm[0, :] = 0
        self.dum, self.dvm = self.fsm.clone(), self.fsm.clone()
        self.dum[:, 0] = 0
        self.dvm[:, -1] = 0
        self.dz = torch.tensor([1. / 15] * 15 + [0.])
        self.zz = -(torch.arange(16, dtype=torch.float32) + .5) / 15
        for field in ('u', 'v', 't', 's', 'rho'):
            setattr(self, field, torch.zeros(nx, ny, 16))
        self.el = torch.zeros(nx, ny)
        self.et = torch.zeros(nx, ny)


def random_fields(seed=0):
    generator = torch.Generator().manual_seed(seed)
    result = {field: torch.randn(8, 6, 16, generator=generator) * .1 for field in FIELDS if field != 'el'}
    result['el'] = torch.randn(8, 6, generator=generator) * .01
    return result


def append_json(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value) + '\n')


def paired_fixture(root, stage, days=2):
    root.mkdir(parents=True)
    case = root / 'm7_day8000'
    case.mkdir()
    (root / 'manifest.json').write_text(json.dumps({'contract': {'core': 'implicit_p0_fixed'},
        'cases': [[7, 8000]], 'days': days, 'stage': stage}))
    for lead in range(1, days + 1):
        append_json(case / 'daily.jsonl', {'lead_day': lead, 'time': 8000. + lead,
                    'metrics': {f'{timing}/mse/{field}': 4. if stage == 'physics' else 1.
                                for timing in ('post',) for field in FIELDS}})
    return root


class FakeData:
    def load_initial(self, member, day, map_location=None):
        fields = random_fields(3)
        fields['et'] = torch.zeros(8, 6)
        fields['time'] = torch.tensor(day - 9e-8, dtype=torch.float64)
        return fields


class FakeRuntime:
    def __init__(self, mode='normal'):
        self.device = torch.device('cpu')
        self.core = DiagnosticCore()
        self.core.time = 8000.
        self.data = FakeData()
        self.mode = mode

    def contract(self):
        return {'core': 'implicit_p0_fixed', 'calibration_id': 'fixture'}


class FakeLogger:
    def log(self, *args): pass
    def image(self, *args): pass
    def artifact(self, *args): pass


class TinyModel(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.))


def fake_rollout(runtime, model, member, start, days, baseline=False, on_intraday=None):
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError('Evaluation model must be frozen and in eval mode')
    if runtime.mode == 'mutate':
        with torch.no_grad():
            model.weight.add_(1.)
    if runtime.mode == 'nonfinite':
        runtime.core.time = start + 1 / 24
        raise FloatingPointError('injected fractional-day numerical failure')
    for day in range(1, days + 1):
        ignored = {'pre': None, 'metrics': {'pre/mse/t': float('nan')}} if runtime.mode == 'bad_pre' else {}
        if on_intraday is not None:
            for hour in range(1, 5):
                lead = (day - 1) * 86400 + hour * 3600
                runtime.core.time = start + lead / 86400.
                fields = random_fields(3)
                on_intraday({'lead_seconds': lead, 'time': runtime.core.time, 'within_day_hour': hour,
                             'post': fields, 'reference': fields, 'metrics': {},
                             'correction_diagnostics': {'t_budget': 0.}, **ignored})
        runtime.core.time = start + day
        fields = random_fields(3)
        yield {'lead_day': day, 'time': float(start + day), 'metrics': {},
               'post': fields, **ignored}


def evaluation_config(root, members=None):
    return {'run_root': str(root), 'frequency': '1h', 'model': {'hidden_dim': 4},
            'split': {'evaluation_members': members or [7], 'evaluation_starts': [8000],
                      'evaluation_start_day': 8000, 'evaluation_end_day': 9000}}


def evaluate_failure_rank(rank, init_file, root, failure):
    dist.init_process_group('gloo', init_method='file://' + init_file, rank=rank, world_size=2,
                            timeout=timedelta(seconds=15))
    runtime = FakeRuntime()
    config = evaluation_config(Path(root) / 'campaign', [6, 7])
    from neuralpom.v1 import evaluation
    original = evaluation._save_fields
    original_append = evaluation._append
    def append(path, value):
        if failure == 'intraday' and rank == 1 and Path(path).name == 'intraday.jsonl':
            raise OSError('injected rank-one intraday write failure')
        return original_append(path, value)
    def save_fields(*args):
        if failure == 'write' and rank == 1:
            raise OSError('injected rank-one field write failure')
        return original(*args)
    def render(*args):
        if failure == 'render':
            raise OSError('injected rank-zero plotting failure')
        return []
    try:
        with patch('neuralpom.v1.evaluation.frozen_rollout', fake_rollout), \
             patch('neuralpom.v1.evaluation._save_fields', save_fields), \
             patch('neuralpom.v1.evaluation._append', append), \
             patch('neuralpom.v1.visualization.render_evaluation', render):
            try:
                evaluate(config, runtime, None, 1, FakeLogger(), baseline=True)
            except RuntimeError as error:
                Path(root, f'rank{rank}.txt').write_text(str(error))
    finally:
        dist.destroy_process_group()


class DiagnosticMathTest(unittest.TestCase):
    def test_spectral_energy_error_cross_closure_and_depth_mean(self):
        core = DiagnosticCore()
        prediction, reference = random_fields(10), random_fields(20)
        spectrum = velocity_spectra(prediction, reference, core.dz, core.dum, core.dvm)
        p, r, e, c = (np.asarray(spectrum[key]) for key in ('prediction', 'reference', 'error', 'cross'))
        np.testing.assert_allclose(e, p + r - 2 * c, atol=1e-16, rtol=1e-12)
        self.assertEqual(p.shape[0], 16)
        self.assertTrue(np.all(e >= 0))
        # Identical vertical layers make each layer equal to the depth mean.
        for field in ('u', 'v'):
            prediction[field] = prediction[field][..., :1].expand(-1, -1, 16)
        spectrum = velocity_spectra(prediction, prediction, core.dz, core.dum, core.dvm)
        np.testing.assert_allclose(spectrum['prediction'][0], spectrum['prediction'][-1], atol=1e-16)
        np.testing.assert_allclose(spectrum['error'], 0., atol=0)
        np.testing.assert_allclose(spectrum['cross'], spectrum['prediction'], atol=1e-16)

    def test_tracer_decomposition_closes_on_nonuniform_reference_volume(self):
        core = DiagnosticCore()
        prediction, reference = random_fields(1), random_fields(2)
        reference['et'] = torch.arange(48, dtype=torch.float32).reshape(8, 6) * .1
        metrics, profiles, _ = state_diagnostics(core, prediction, reference)
        volume = (core.art * core.fsm * (core.h + reference['et']))[..., None] * core.dz[:15]
        for field in ('t', 's'):
            error = prediction[field][..., :15] - reference[field][..., :15]
            expected = float((error.double().square() * volume.double()).sum() / volume.sum())
            self.assertAlmostEqual(metrics['mse/' + field], expected, places=8)
            terms = [metrics[f'decomposition/{field}/{name}'] for name in ('global_mean', 'profile_shape', 'horizontal')]
            self.assertAlmostEqual(sum(terms), metrics['mse/' + field], places=13)
            self.assertTrue(all(value >= 0 for value in terms))
            self.assertEqual(len(profiles[f'{field}/prediction']), 15)
        self.assertGreater(metrics['density/mse'], 0.)

    def test_identical_reference_has_zero_field_density_and_spectral_error(self):
        core = DiagnosticCore()
        fields = random_fields(3)
        metrics, profiles, spectrum = state_diagnostics(core, fields, fields)
        for field in FIELDS:
            self.assertEqual(metrics[f'mse/{field}'], 0.)
            self.assertEqual(metrics[f'bias/{field}'], 0.)
        self.assertEqual(metrics['density/mse'], 0.)
        np.testing.assert_array_equal(profiles['rho/prediction'], profiles['rho/reference'])
        np.testing.assert_array_equal(np.asarray(spectrum['error']), 0.)

    def test_uniform_tracer_offset_is_entirely_global_mean_error(self):
        core = DiagnosticCore()
        reference = {field: getattr(core, field).clone() for field in FIELDS}
        prediction = {field: value.clone() for field, value in reference.items()}
        prediction['t'] += 2.
        metrics, _, _ = state_diagnostics(core, prediction, reference)
        self.assertAlmostEqual(metrics['mse/t'], 4., places=12)
        self.assertAlmostEqual(metrics['decomposition/t/global_mean'], 4., places=12)
        self.assertAlmostEqual(metrics['decomposition/t/profile_shape'], 0., places=12)
        self.assertAlmostEqual(metrics['decomposition/t/horizontal'], 0., places=12)

    def test_nonfinite_prediction_is_rejected(self):
        core = DiagnosticCore()
        prediction, reference = random_fields(1), random_fields(2)
        prediction['t'][1, 1, 1] = torch.nan
        with self.assertRaises(FloatingPointError):
            state_diagnostics(core, prediction, reference)


class EvaluationTest(unittest.TestCase):
    def test_actual_evaluate_frozen_fields_geometry_and_reference_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = FakeRuntime()
            weights = root / 'weights.pt'
            model = TinyModel()
            torch.save({'stage': 'posttrain', 'contract': runtime.contract(), 'model': model.state_dict()}, weights)
            with patch('neuralpom.v1.evaluation.CorrectorNet', TinyModel), \
                 patch('neuralpom.v1.evaluation.frozen_rollout', fake_rollout), \
                 patch('neuralpom.v1.visualization.render_evaluation', return_value=[]):
                output = evaluate(evaluation_config(root), runtime, weights, 1, FakeLogger())
            self.assertTrue((output / 'COMPLETE.json').exists())
            self.assertFalse(list(output.rglob('*_pre.npz')))
            for filename in ('manifest.json', 'summary.json'):
                self.assertEqual(json.loads((output / filename).read_text())['evaluation_field'], 'post_correction')
            for filename in ('profiles.jsonl', 'spectra.jsonl'):
                self.assertEqual(set(json.loads((output / 'm7_day8000' / filename).read_text())), {'lead_day', 'post'})
            intraday = [json.loads(line) for line in (output / 'm7_day8000/intraday.jsonl').read_text().splitlines()]
            self.assertEqual([row['lead_seconds'] for row in intraday], [3600, 7200, 10800, 14400])
            self.assertEqual([row['within_day_hour'] for row in intraday], [1, 2, 3, 4])
            self.assertTrue(all(row['metrics']['post/mse/t'] == 0. for row in intraday))
            with np.load(output / 'geometry.npz') as archive:
                self.assertIn('fsm', archive.files)
                self.assertIn('dum', archive.files)
                self.assertIn('dvm', archive.files)
            with np.load(output / 'm7_day8000/day001_reference.npz') as archive:
                self.assertAlmostEqual(float(archive['time']), 8001. - 9e-8, places=11)
            self.assertEqual(float(torch.load(weights, weights_only=False)['model']['weight']), 2.)
            self.assertTrue(torch.equal(runtime.core.t, torch.zeros_like(runtime.core.t)))

    def test_legacy_pre_metrics_never_enter_new_evaluation_or_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('neuralpom.v1.evaluation.frozen_rollout', fake_rollout), \
                 patch('neuralpom.v1.visualization.render_evaluation', return_value=[]):
                output = evaluate(evaluation_config(Path(directory)), FakeRuntime('bad_pre'), None, 1,
                                  FakeLogger(), baseline=True)
            for name in ('daily.jsonl', 'intraday.jsonl'):
                rows = [json.loads(line) for line in (output / 'm7_day8000' / name).read_text().splitlines()]
                self.assertTrue(all(all(key.startswith('post/') for key in row['metrics']) for row in rows))
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['completed_cases'], 1)
            self.assertEqual(summary['periods']['whole']['post/rmse/t'], 0.)
            self.assertNotIn('pre/', json.dumps(summary))

    def test_evaluate_detects_weight_mutation_and_wrong_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = FakeRuntime('mutate')
            weights = root / 'weights.pt'
            torch.save({'stage': 'posttrain', 'contract': runtime.contract(), 'model': TinyModel().state_dict()}, weights)
            with patch('neuralpom.v1.evaluation.CorrectorNet', TinyModel), \
                 patch('neuralpom.v1.evaluation.frozen_rollout', fake_rollout):
                with self.assertRaisesRegex(RuntimeError, 'modified model'):
                    evaluate(evaluation_config(root), runtime, weights, 1, FakeLogger())
            torch.save({'stage': 'posttrain', 'contract': {'core': 'implicit'}, 'model': {}}, weights)
            with self.assertRaisesRegex(ValueError, 'contract'):
                evaluate(evaluation_config(root / 'other'), runtime, weights, 1, FakeLogger())

    def test_prefix_weight_metadata_reaches_manifest_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = FakeRuntime()
            weights = root / 'posttrain_prefix.pt'
            provenance = {'selection': 'validation_prefix', 'step': 2, 'validated_windows': 2}
            torch.save({'stage': 'posttrain', 'contract': runtime.contract(),
                        'model': TinyModel().state_dict(), **provenance}, weights)
            with patch('neuralpom.v1.evaluation.CorrectorNet', TinyModel), \
                 patch('neuralpom.v1.evaluation.frozen_rollout', fake_rollout), \
                 patch('neuralpom.v1.visualization.render_evaluation', return_value=[]):
                model_output = evaluate(evaluation_config(root), runtime, weights, 1, FakeLogger())
                baseline_output = evaluate(evaluation_config(root), runtime, None, 1, FakeLogger(), baseline=True)
            self.assertEqual(model_output.name, 'posttrain_1d')
            for filename in ('manifest.json', 'summary.json'):
                model = json.loads((model_output / filename).read_text())
                baseline = json.loads((baseline_output / filename).read_text())
                self.assertEqual({key: model[key] for key in provenance}, provenance)
                self.assertTrue(all(baseline[key] is None for key in provenance))
            comparison = compare_to_physics(model_output, baseline_output, root / 'comparison.json')
            paired = json.loads(comparison.read_text())
            self.assertEqual({key: paired[key] for key in provenance}, provenance)
            from matplotlib.figure import Figure
            original_title, titles = Figure.suptitle, []
            def title(figure, text, *args, **kwargs):
                titles.append(text)
                return original_title(figure, text, *args, **kwargs)
            with patch.object(Figure, 'suptitle', title):
                render_comparison(comparison)
            self.assertIn('Validation prefix (2 windows)', titles[0])

    def test_evaluate_nonfinite_records_fractional_anomaly_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('neuralpom.v1.evaluation.frozen_rollout', fake_rollout), \
                 patch('neuralpom.v1.visualization.render_evaluation', return_value=[]):
                output = evaluate(evaluation_config(root), FakeRuntime('nonfinite'), None, 1, FakeLogger(), baseline=True)
            status = json.loads((output / 'm7_day8000/status.json').read_text())
            self.assertEqual(status['status'], 'nonfinite')
            self.assertAlmostEqual(status['first_anomaly_lead_seconds'], 3600., places=6)
            self.assertFalse((output / 'COMPLETE.json').exists())
            self.assertEqual(json.loads((output / 'summary.json').read_text())['completion_fraction'], 0.)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo unavailable')
    def test_actual_evaluate_distributes_rank_io_and_plot_failures(self):
        for failure in ('write', 'render', 'intraday'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                context = mp.spawn(evaluate_failure_rank,
                    args=(str(Path(directory) / 'rendezvous'), directory, failure), nprocs=2, join=False)
                import time
                deadline = time.monotonic() + 25
                while time.monotonic() < deadline and not context.join(timeout=1):
                    pass
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                        self.fail('Evaluation failure left a peer waiting')
                for rank in range(2):
                    message = (Path(directory) / f'rank{rank}.txt').read_text()
                    self.assertIn('Distributed evaluation failed', message)
                    self.assertIn('injected rank-', message)

    def test_paired_skill_and_missing_lead_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = paired_fixture(root / 'model', 'posttrain')
            physics = paired_fixture(root / 'physics', 'physics')
            output = compare_to_physics(model, physics, root / 'comparison.json')
            result = json.loads(output.read_text())
            self.assertEqual(len(result['daily']), 2)
            self.assertEqual(result['daily'][0]['rmse_ratio']['post/t'], .5)
            physical_file = physics / 'm7_day8000/daily.jsonl'
            physical_file.write_text(physical_file.read_text().splitlines()[0] + '\n')
            with self.assertRaises(ValueError):
                compare_to_physics(model, physics, root / 'bad.json')

    def test_paired_records_with_wrong_time_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = paired_fixture(root / 'model', 'posttrain')
            physics = paired_fixture(root / 'physics', 'physics')
            path = physics / 'm7_day8000/daily.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]['time'] += 1 / 24
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            with self.assertRaisesRegex(ValueError, '(times|clock)'):
                compare_to_physics(model, physics, root / 'bad.json')

    def test_legacy_pre_metrics_do_not_affect_summary_or_paired_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = paired_fixture(root / 'model', 'posttrain')
            physics = paired_fixture(root / 'physics', 'physics')
            for output in (model, physics):
                path = output / 'm7_day8000/daily.jsonl'
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                for row in rows:
                    row['metrics'].update({f'pre/mse/{field}': float('nan') for field in FIELDS})
                path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            result = json.loads(compare_to_physics(model, physics, root / 'paired.json').read_text())
            self.assertEqual(result['evaluation_field'], 'post_correction')
            self.assertEqual(result['periods']['whole']['rmse_ratio']['post/t'], .5)
            self.assertNotIn('pre/', json.dumps(result))
            summary = summarize_cases([{'status': 'complete', 'completed_days': 2,
                                        'metrics_file': str(model / 'm7_day8000/daily.jsonl')}], 2)
            self.assertEqual(summary['periods']['whole']['post/rmse/t'], 1.)
            self.assertNotIn('pre/', json.dumps(summary))

    def test_paired_period_rmse_uses_same_cases_and_leads_before_ratio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = [[6, 8000], [7, 8064]]
            days = 60
            for name, stage in (('model', 'posttrain'), ('physics', 'physics')):
                path = root / name
                path.mkdir()
                (path / 'manifest.json').write_text(json.dumps({'contract': {'core': 'implicit_p0_fixed'},
                    'cases': cases, 'days': days, 'stage': stage}))
                for case_index, (member, start) in enumerate(cases):
                    case = path / f'm{member}_day{start:04d}'
                    case.mkdir()
                    for lead in range(1, days + 1):
                        value = float(lead + case_index * 10) if name == 'model' else float(lead**2 + case_index * 100)
                        append_json(case / 'daily.jsonl', {'lead_day': lead, 'time': float(start + lead),
                            'metrics': {f'{timing}/mse/{field}': value
                                        for timing in ('post',) for field in FIELDS}})
            output = compare_to_physics(root / 'model', root / 'physics', root / 'paired.json')
            result = json.loads(output.read_text())
            for name, lower, upper in (('first50', 1, 50), ('whole', 1, 60), ('last50', 11, 60)):
                period = result['periods'][name]
                model_values = [lead + case * 10 for case in range(2) for lead in range(lower, upper + 1)]
                baseline_values = [lead**2 + case * 100 for case in range(2) for lead in range(lower, upper + 1)]
                expected_model = np.sqrt(np.mean(model_values))
                expected_baseline = np.sqrt(np.mean(baseline_values))
                self.assertAlmostEqual(period['model_rmse']['post/t'], expected_model)
                self.assertAlmostEqual(period['physics_rmse']['post/t'], expected_baseline)
                self.assertAlmostEqual(period['rmse_ratio']['post/t'], expected_model / expected_baseline)
                self.assertEqual(period['paired_case_leads'], 2 * (upper - lower + 1))
                self.assertEqual(period['case_count'], 2)
                self.assertEqual(len(period['per_case']), 2)
                for case in period['per_case']:
                    self.assertEqual(case['paired_case_leads'], upper - lower + 1)
            daily_average = np.mean([row['rmse_ratio']['post/t'] for row in result['daily']])
            self.assertNotAlmostEqual(result['periods']['whole']['rmse_ratio']['post/t'], daily_average)

    def test_zero_baseline_has_null_daily_and_aggregate_ratios(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = paired_fixture(root / 'model', 'posttrain')
            physics = paired_fixture(root / 'physics', 'physics')
            path = physics / 'm7_day8000/daily.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            for row in rows:
                row['metrics'] = {key: 0. for key in row['metrics']}
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            output = compare_to_physics(model, physics, root / 'paired.json')
            result = json.loads(output.read_text())
            figure = render_comparison(output)
            self.assertEqual(figure.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
            self.assertTrue(all(row['rmse_ratio']['post/t'] is None for row in result['daily']))
            for period in result['periods'].values():
                self.assertIsNone(period['rmse_ratio']['post/t'])
                self.assertIsNone(period['per_case'][0]['rmse_ratio']['post/t'])
                self.assertEqual(period['physics_rmse']['post/t'], 0.)
                self.assertEqual(period['model_rmse']['post/t'], 1.)

    def test_missing_baseline_case_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = paired_fixture(root / 'model', 'posttrain')
            physics = paired_fixture(root / 'physics', 'physics')
            (physics / 'm7_day8000/daily.jsonl').unlink()
            with self.assertRaises((ValueError, FileNotFoundError)):
                compare_to_physics(model, physics, root / 'bad.json')

    def test_partial_cases_excluded_from_skill_but_retained_in_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = paired_fixture(root / 'model', 'posttrain')
            complete = {'status': 'complete', 'completed_days': 2,
                        'metrics_file': str(model / 'm7_day8000/daily.jsonl')}
            failed = {'status': 'nonfinite', 'completed_days': 0, 'metrics_file': str(root / 'missing')}
            summary = summarize_cases([complete, failed], 2)
            self.assertEqual(summary['completion_fraction'], .5)
            self.assertEqual(summary['periods']['whole']['post/rmse/t'], 1.)
            self.assertEqual(len(summary['cases']), 2)

    def test_render_complete_diagnostic_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = root / 'm7_day8000'
            case.mkdir()
            core = DiagnosticCore()
            reference, prediction = random_fields(1), random_fields(2)
            metrics, profile, spectrum = state_diagnostics(core, prediction, reference)
            append_json(case / 'daily.jsonl', {'lead_day': 1, 'time': 8001.,
                'metrics': {f'{timing}/{key}': value for timing in ('post',) for key, value in metrics.items()}})
            append_json(case / 'profiles.jsonl', {'lead_day': 1, 'post': profile})
            append_json(case / 'spectra.jsonl', {'lead_day': 1, 'post': spectrum})
            np.savez(root / 'geometry.npz', fsm=core.fsm.numpy(), dum=core.dum.numpy(), dvm=core.dvm.numpy())
            for hour in range(1, 5):
                append_json(case / 'intraday.jsonl', {'lead_seconds': hour * 3600, 'within_day_hour': hour,
                    'time': 8000 + hour / 24,
                    'metrics': {f'{timing}/{key}': value for timing in ('post',) for key, value in metrics.items()}})
            for name, state in (('reference', reference), ('post', prediction)):
                np.savez(case / f'day001_{name}.npz', **{key: value.numpy() for key, value in state.items()})
            (root / 'summary.json').write_text(json.dumps({'requested_cases': 1, 'completed_cases': 1,
                'cases': [{'member': 7, 'start_day': 8000, 'status': 'complete', 'completed_days': 1,
                           'metrics_file': str(case / 'daily.jsonl')}]}))
            images = render_evaluation(root)
            self.assertGreaterEqual(len(images), 4)
            for required in ('daily_rmse', 'intraday_rmse', '_profiles', '_spectra', '_fields'):
                self.assertTrue(any(required in image.stem for image in images))
            for image in images:
                self.assertTrue(image.exists())
                self.assertGreater(image.stat().st_size, 1000)
                self.assertEqual(image.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
            from matplotlib.figure import Figure
            from neuralpom.v1.visualization import _prefix_suffix
            self.assertEqual(_prefix_suffix({'selection': 'final'}), '')
            summary = json.loads((root / 'summary.json').read_text())
            summary.update(selection='validation_prefix', step=2, validated_windows=2)
            (root / 'summary.json').write_text(json.dumps(summary))
            original_title, titles = Figure.suptitle, []
            def title(figure, text, *args, **kwargs):
                titles.append(text)
                return original_title(figure, text, *args, **kwargs)
            with patch.object(Figure, 'suptitle', title):
                render_evaluation(root)
            self.assertEqual(len(titles), len(images))
            self.assertTrue(all('Validation prefix (2 windows)' in value for value in titles))
            # Historical files may retain arbitrary pre data. Re-reporting
            # reads only post/reference, including the surface color limits.
            for name in ('daily.jsonl', 'intraday.jsonl'):
                path = case / name
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                for row in rows:
                    row['metrics'].update({f'pre/rmse/{field}': 1e30 for field in FIELDS})
                path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            for name in ('profiles.jsonl', 'spectra.jsonl'):
                path = case / name
                row = json.loads(path.read_text())
                row['pre'] = 'invalid obsolete diagnostics'
                path.write_text(json.dumps(row) + '\n')
            (case / 'day001_pre.npz').write_bytes(b'invalid obsolete field')
            original_save, figures = Figure.savefig, {}
            def save(figure, path, *args, **kwargs):
                figures[Path(path).stem] = figure
                return original_save(figure, path, *args, **kwargs)
            with patch.object(Figure, 'savefig', save):
                render_evaluation(root)
            for axis, field in zip(figures['daily_rmse'].axes, FIELDS):
                self.assertEqual(len(axis.lines), 1)
                np.testing.assert_array_equal(axis.lines[0].get_ydata(), [metrics[f'rmse/{field}']])
            profile_figure = figures['m7_d8000_lead1_profiles']
            self.assertTrue(all(len(axis.lines) == 2 for axis in profile_figure.axes))
            fields_figure = figures['m7_d8000_lead1_fields']
            self.assertEqual(sum(len(axis.images) for axis in fields_figure.axes), 3 * len(FIELDS))
            self.assertFalse(any('pre:' in axis.get_title() for axis in fields_figure.axes))


if __name__ == '__main__':
    unittest.main()

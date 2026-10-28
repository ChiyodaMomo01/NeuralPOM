import unittest
from pathlib import Path
from neuralpom.v1.config import load_config, resolve_config
from neuralpom.v1.sampling import STAGES, stage_plan, stage_samples


class SamplingTests(unittest.TestCase):
    def test_production_frequency_budgets_and_cache_coverage(self):
        root = Path(__file__).resolve().parents[1]
        for frequency, stride, samples, updates, epochs in (
                ('1h', 24, 1750, 219, 10), ('1day', 72, 588, 74, 3)):
            config = load_config(root / 'configs' / 'v1' / f'{frequency}.json')
            plan = stage_plan(config, 'midtrain4')
            self.assertEqual((plan['stride_days'], plan['samples'],
                              plan['updates_per_epoch'], plan['epochs']),
                             (stride, samples, updates, epochs))
            self.assertEqual(plan['maximum_updates'], updates * epochs)
            self.assertEqual(plan['rollout_steps'], 4)
            self.assertEqual(plan['gradient_method'], 'full_bptt')
            cases = stage_samples(config, 'midtrain4')
            self.assertTrue(set(cases) <= set(stage_samples(config, 'pretrain')))
            self.assertEqual(stage_plan(config, 'pretrain')['maximum_updates'], 35000)
            self.assertEqual(stage_plan(config, 'posttrain')['maximum_updates'], 17600)
            if frequency == '1day':
                self.assertEqual([d for m, d in cases if m == 0], list(range(1000, 7000, 72)))
                self.assertEqual(max(d for _, d in cases), 6976)
                self.assertEqual(plan['padded_samples_per_epoch'], 4)

    def test_registered_budgets_and_model_time_labels(self):
        for frequency in ('1h', '1day'):
            config = resolve_config({'frequency': frequency, 'data_root': '/data/fixed', 'run_root': '/runs/fixed'})
            expected = [('pretrain', 7000, 875, 35000), ('midtrain4', 1750, 219, 2190),
                        ('posttrain', 700, 880, 17600)]
            self.assertEqual(tuple(item[0] for item in expected), STAGES)
            pretrain = set(stage_samples(config, 'pretrain'))
            for stage, samples, epoch_updates, total in expected:
                plan = stage_plan(config, stage)
                self.assertEqual((plan['samples'], plan['updates_per_epoch'], plan['maximum_updates']),
                                 (samples, epoch_updates, total))
                cases = stage_samples(config, stage)
                self.assertEqual(len(cases), len(set(cases)))
                self.assertTrue(set(cases) <= pretrain)
                self.assertTrue(all(day * 86400 + plan['rollout_steps'] * plan['interval_seconds'] < 7000 * 86400 for _, day in cases))

    def test_retired_curriculum_cannot_enter_training_plan(self):
        config = resolve_config({'data_root': '/data/fixed', 'run_root': '/runs/fixed'})
        for stage in ('midtrain2', 'midtrain3'):
            with self.assertRaises(ValueError):
                stage_plan(config, stage)
        with self.assertRaisesRegex(ValueError, 'single 4-step'):
            resolve_config({'data_root': '/data/fixed', 'run_root': '/runs/fixed',
                            'stages': {'midtrain': {'horizons': [2, 3, 4]}}})

    def test_old_core_and_disabled_tracking_rejected(self):
        for override in ({'core': 'implicit'}, {'wandb': {'mode': 'disabled'}},
                         {'loss': {'components': [{'name': 'spectral'}]}}):
            with self.assertRaises(ValueError):
                resolve_config({'data_root': '/data/fixed', 'run_root': '/runs/fixed', **override})

    def test_post_window_budget_matches_registered_hours(self):
        for frequency in ('1day',):
            config = resolve_config({'frequency': frequency, 'data_root': '/data/fixed', 'run_root': '/runs/fixed',
                                     'stages': {'posttrain': {'tbptt_hours': 48}}})
            plan = stage_plan(config, 'posttrain')
            self.assertEqual((plan['updates_per_epoch'], plan['maximum_updates'], plan['tbptt_seconds']),
                             (440, 8800, 172800))
        for hours in (0, 25, 72):
            with self.assertRaisesRegex(ValueError, 'whole days'):
                resolve_config({'data_root': '/data/fixed', 'run_root': '/runs/fixed',
                                'stages': {'posttrain': {'tbptt_hours': hours}}})

    def test_hourly_post_gradient_policy_is_explicit_and_daily_updates_required(self):
        base = {'frequency': '1h', 'data_root': '/data/fixed', 'run_root': '/runs/fixed'}
        config = resolve_config(base)
        self.assertEqual(config['stages']['posttrain']['gradient_mode'], 'last_correction_only')
        self.assertEqual(stage_plan(config, 'posttrain')['gradient_method'], 'last_correction_only')
        for override in ({'tbptt_hours': 48}, {'gradient_mode': 'full_window'}):
            with self.assertRaises(ValueError):
                resolve_config({**base, 'stages': {'posttrain': override}})

    def test_split_leakage_rejected(self):
        with self.assertRaises(ValueError):
            resolve_config({'data_root': '/data/fixed', 'run_root': '/runs/fixed',
                            'split': {'train_members': [0, 7]}})

    def test_strict_kernel_metadata_matches_executed_operator(self):
        for mode in ('strict_vertical', 'strict_vertical_compiled'):
            with self.assertRaisesRegex(ValueError, 'only proft'):
                resolve_config({'data_root': '/data/fixed', 'run_root': '/runs/fixed',
                                'acceleration': {'mode': mode, 'compiled_kernels': ['profq']}})


if __name__ == '__main__':
    unittest.main()

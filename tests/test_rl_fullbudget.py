"""Full-budget regression: worsening validation cannot stop disabled runs."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from neuralpom.v1.posttrain.rl import RLPosttrain
from neuralpom.v1.posttrain.policy import validate_settings, early_stop_reached
from neuralpom.v1.config import resolve_config
from neuralpom.v1.sampling import stage_plan

class FullBudgetChecks(unittest.TestCase):
    def test_budget_matches_both_original_ddp_plans(self):
        for frequency in ('1day', '1h'):
            cfg = resolve_config({
                'frequency': frequency, 'data_root': './data', 'run_root': './runs/test',
                'stages': {'posttrain': {
                    'method': 'rl',
                    'rl': {
                        'contexts_per_batch': 2, 'group_size': 4, 'noise_std': .03,
                        'failure_daily_loss': 1000., 'max_trajectories': 14080,
                        'validation_every_trajectories': 128, 'patience': None,
                    },
                }},
            })
            plan = stage_plan(cfg, 'posttrain')
            expected = plan['trajectory_batches_per_epoch'] * cfg['world_size'] * plan['epochs']
            self.assertEqual(expected, 14080)
            self.assertEqual(cfg['stages']['posttrain']['rl']['max_trajectories'], expected)
            self.assertEqual(expected // cfg['world_size'], 1760)
            self.assertEqual(plan['maximum_updates'], 17600)
            validate_settings(cfg['stages']['posttrain']['rl'], 8, 10)
            self.assertFalse(early_stop_reached(cfg['stages']['posttrain']['rl'], 100000))

    def test_worsening_validation_preserves_best_without_stopping(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = RLPosttrain.__new__(RLPosttrain)
            trainer.rank, trainer.world, trainer.step = 0, 1, 0
            trainer.model = torch.nn.Linear(1, 1)
            trainer.runtime = SimpleNamespace(device=torch.device('cpu'))
            trainer.validation_cases = [(0, 0)]
            trainer.options = {'patience': None, 'failure_daily_loss': 1000., 'min_delta': 0.}
            trainer.config = {'run_root': directory}
            trainer.output = Path(directory)
            trainer.best, trainer.best_step, trainer.bad_checks = float('inf'), 0, 0
            trainer.stopped, trainer.initialized = False, False
            saved = []
            trainer.weights = lambda path, selection, **kwargs: saved.append((path.name, selection, kwargs))
            trainer.log = lambda row: None
            trainer.save = lambda: None
            for step, loss in enumerate([1., 2., 3., 4., 5., 6., 7.]):
                trainer.step = step
                episode = SimpleNamespace(daily_losses=torch.tensor([loss]), failed=False, physical_steps=10)
                with patch('neuralpom.v1.posttrain.rl.collect', return_value=episode):
                    trainer.validate()
            self.assertEqual(trainer.bad_checks, 6)
            self.assertFalse(trainer.stopped)
            self.assertEqual(trainer.best, 1.)
            self.assertEqual(trainer.best_step, 0)
            self.assertEqual(trainer.latest_validation, 7.)
            self.assertTrue(any(name == 'posttrain_best.pt' for name, _, _ in saved))
            self.assertFalse(any(name == 'posttrain.pt' for name, _, _ in saved))

if __name__ == '__main__':
    unittest.main(verbosity=2)

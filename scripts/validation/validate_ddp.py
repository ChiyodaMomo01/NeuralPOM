"""Eight-rank, hidden-256 validation for either correction frequency.

The two post-train windows are a distributed correctness prefix, not a
completed ten-day post-train run. Configuration controls the physics backend.
"""
from pathlib import Path
import argparse
import sys
import json
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import numpy as np
import torch
import torch.distributed as dist
from neuralpom.v1.cache import build_assets
from neuralpom.v1.checkpoint import atomic_json, atomic_torch, cpu_tree, unwrap, rng_state
from neuralpom.v1.cli import initialize, create_logger, collective_call
from neuralpom.v1.config import load_config
from neuralpom.v1.engine import Training, rank_zero_action, read_records
from neuralpom.v1.runtime import Runtime
from neuralpom.v1.sampling import STAGES, stage_samples, post_windows


class ValidationBoundary(Exception):
    pass


class PrefixTraining(Training):
    stop_at = 1

    def record_timing(self, *args):
        super().record_timing(*args)
        if self.step == self.stop_at:
            raise ValidationBoundary()


def validate_configuration(config, frequency=None):
    if config['world_size'] != 8 or config['model']['hidden_dim'] != 256:
        raise ValueError('Distributed validation requires 8 ranks and hidden_dim 256')
    if frequency is not None and config['frequency'] != frequency:
        raise ValueError('--frequency must match the recorded JSON configuration')
    if any(config['stages'][stage]['checkpoint_every_updates'] != 1
           for stage in ('pretrain', 'midtrain', 'posttrain')):
        raise ValueError('DDP validation requires checkpoint_every_updates=1 for every stage')
    if post_windows(config) <= 2:
        raise ValueError('The two-window prefix must be shorter than the registered rollout')


def assert_restart(training, expected, device):
    assert training.step == 1 and training.cursor == expected['cursor']
    torch.testing.assert_close(cpu_tree(unwrap(training.model).state_dict()), expected['weights'], rtol=0, atol=0)
    optimizer = cpu_tree(training.optimizer.state_dict())
    assert optimizer['param_groups'] == expected['optimizer']['param_groups']
    torch.testing.assert_close(optimizer['state'], expected['optimizer']['state'], rtol=0, atol=0)
    physical = training.physical
    assert physical['scalars'] == expected['physical']['scalars']
    torch.testing.assert_close(physical['tensors'], expected['physical']['tensors'], rtol=0, atol=0)
    assert physical['objects'].keys() == expected['physical']['objects'].keys()
    for key, value in expected['physical']['objects'].items():
        restored = physical['objects'][key]
        assert type(restored) is type(value)
        assert vars(restored) == vars(value) if hasattr(value, '__dict__') else restored == value
    actual_rng = rng_state(device)
    for key in ('torch', 'cuda'):
        torch.testing.assert_close(actual_rng[key], expected['rng'][key], rtol=0, atol=0)
    assert actual_rng['python'] == expected['rng']['python']
    for actual, original in zip(actual_rng['numpy'], expected['rng']['numpy']):
        np.testing.assert_array_equal(actual, original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--frequency', choices=('1h', '1day'), help='Assert the frequency recorded in --config')
    args = parser.parse_args()
    config = load_config(args.config)
    validate_configuration(config, args.frequency)
    device, rank = initialize(config)
    logger = None
    report = {'world_size': 8, 'hidden_dim': 256, 'frequency': config['frequency'],
              'acceleration': config['acceleration'], 'checkpoint_every_updates': 1, 'stages': {}}
    try:
        logger = create_logger(config, 'prepare', rank)
        runtime = collective_call(lambda: Runtime(config, device, rank, assets=False))
        build_assets(config, runtime.core, runtime.reset, lambda seconds: runtime.advance(False, seconds),
                     stage_samples(config, 'pretrain'), stage_samples(config, 'pretrain', 'validation'), logger)
        collective_call(lambda: logger.finish(True))
        logger = None
        for stage in STAGES:
            logger = create_logger(config, stage, rank)
            runtime = collective_call(lambda: Runtime(config, device, rank))
            if stage != 'posttrain':
                training = Training(config, stage, runtime, logger)
                report['stages'][stage] = {'updates': training.train(), 'complete': True}
            else:
                training = PrefixTraining(config, stage, runtime, logger)
                try:
                    training.train()
                    raise AssertionError('Expected first-window restart boundary')
                except ValidationBoundary:
                    expected = {'weights': cpu_tree(unwrap(training.model).state_dict()),
                        'optimizer': cpu_tree(training.optimizer.state_dict()),
                        'physical': runtime.snapshot(), 'rng': rng_state(device), 'cursor': dict(training.cursor)}
                del training
                runtime = collective_call(lambda: Runtime(config, device, rank))
                training = PrefixTraining(config, stage, runtime, logger)
                collective_call(lambda: assert_restart(training, expected, device))
                del expected
                training.stop_at = 2
                try:
                    training.train()
                    raise AssertionError('Expected second-window validation boundary')
                except ValidationBoundary:
                    report['stages'][stage] = {'validated_windows': 2, 'full_rollout_complete': False,
                        'restart_weights_exact': True, 'restart_optimizer_exact': True,
                        'restart_physical_exact': True, 'restart_rng_exact': True, 'cursor': training.cursor}
                def freeze_prefix():
                    training.resource_summary()
                    path = Path(config['run_root']) / 'frozen' / 'posttrain_prefix.pt'
                    atomic_torch(path, {'model': cpu_tree(unwrap(training.model).state_dict()),
                        'contract': runtime.contract(), 'stage': stage, 'step': training.step,
                        'posttrain_gradient_mode': config['stages']['posttrain']['gradient_mode'],
                                'supervision': config['supervision'], 'selection': 'validation_prefix', 'validated_windows': 2})
                    logger.artifact(path, 'posttrain-validation-prefix', 'model')
                    report['stages'][stage]['weights'] = str(path)
                rank_zero_action(freeze_prefix)
            def record_stage():
                pointer = json.loads((training.output / 'resume.json').read_text())
                path = training.output / pointer['checkpoint']
                generation_bytes = sum(item.stat().st_size for item in path.parent.glob('*.pt'))
                entry = report['stages'][stage]
                entry['checkpoint_generation_bytes'] = generation_bytes
                entry['timings'] = read_records(training.output / 'timing.jsonl')
                entry['resources'] = json.loads((training.output / 'resources.json').read_text())
                online_steps = [int(row['online_forward_physics_steps']) for row in read_records(training.output / 'train.jsonl')]
                if stage == 'posttrain':
                    width = config['stages']['posttrain']['tbptt_hours'] * 3600
                    assert online_steps == [(width - config['interval_seconds']) // 600, width // 600]
                else:
                    expected_steps = (training.plan['rollout_steps'] - 1) * config['interval_seconds'] // 600
                    assert all(value == expected_steps for value in online_steps)
                entry['online_forward_physics_steps_per_rank'] = online_steps
                logger.log({'checkpoint/generation_bytes': generation_bytes}, training.step)
                atomic_json(Path(config['run_root']) / 'ddp_validation.json', report)
            rank_zero_action(record_stage)
            if stage == 'posttrain':
                rank_zero_action(lambda: logger.log({'validation/post_windows': 2,
                    'validation/full_post_rollout_complete': False,
                    'validation/restart_weights_exact': True, 'validation/restart_optimizer_exact': True,
                    'validation/restart_physical_exact': True, 'validation/restart_rng_exact': True}, training.step))
            rank_zero_action(lambda: logger.artifact(Path(config['run_root']) / 'ddp_validation.json',
                                                     'ddp-' + stage + '-validation', 'report'))
            collective_call(lambda: logger.finish(True))
            logger = None
            del training
        rank_zero_action(lambda: atomic_json(Path(config['run_root']) / 'DDP_VALIDATION_PASSED.json', report))
    finally:
        if logger is not None:
            logger.finish(False)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()

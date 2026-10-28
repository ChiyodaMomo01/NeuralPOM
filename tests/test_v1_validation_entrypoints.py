"""Validate both physical horizons and complete restart checks in the DDP gate."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from neuralpom.v1.checkpoint import cpu_tree, rng_state
from neuralpom.v1.config import load_config, resolve_config
from neuralpom.v1.sampling import stage_plan

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'validation' / 'validate_ddp.py'
spec = importlib.util.spec_from_file_location('validation_ddp_entrypoint', SCRIPT)
entrypoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entrypoint)


@pytest.mark.parametrize('frequency, interval', [('1h', 3600), ('1day', 86400)])
def test_ddp_entrypoint_accepts_both_frequencies_without_shortening_daily_horizons(tmp_path, frequency, interval):
    config = resolve_config({'data_root': str(tmp_path), 'run_root': str(tmp_path),
        'frequency': frequency, 'world_size': 8, 'model': {'hidden_dim': 256},
        'stages': {stage: {'checkpoint_every_updates': 1} for stage in ('pretrain', 'midtrain', 'posttrain')}})
    entrypoint.validate_configuration(config, frequency)
    for horizon in (4,):
        plan = stage_plan(config, f'midtrain{horizon}')
        assert plan['rollout_steps'] * plan['interval_seconds'] == horizon * interval
        assert plan['gradient_method'] == 'full_bptt'
        assert plan['tbptt_seconds'] is None
    with pytest.raises(ValueError, match='match'):
        entrypoint.validate_configuration(config, '1day' if frequency == '1h' else '1h')


def test_prepared_daily_validation_config_has_one_update_per_mid_stage():
    config = load_config(SCRIPT.parent / 'configs' / 'strict_1day_ddp8.json')
    entrypoint.validate_configuration(config, '1day')
    assert config['acceleration']['mode'] == 'strict_vertical_compiled'
    assert config['acceleration']['cg_backend'] == 'triton'
    for stage, samples in {'pretrain': 8, 'midtrain4': 2}.items():
        plan = stage_plan(config, stage)
        assert plan['samples'] == samples
        assert plan['maximum_updates'] == 1


def test_restart_verification_checks_cloned_policies_optimizer_and_all_rng_states():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 1)).sum().backward()
    optimizer.step()
    physical = {'scalars': {'time': 1002.}, 'tensors': {'u': torch.tensor([1.])},
                'objects': {'policy': SimpleNamespace(name='fixed')}}
    training = SimpleNamespace(step=1, cursor={'epoch': 0, 'batch': 0, 'window': 1},
                               model=model, optimizer=optimizer, physical=physical)
    expected = {'cursor': dict(training.cursor), 'weights': cpu_tree(model.state_dict()),
                'optimizer': cpu_tree(optimizer.state_dict()), 'physical': copy.deepcopy(physical),
                'rng': rng_state('cpu')}
    entrypoint.assert_restart(training, expected, 'cpu')
    training.physical['tensors']['u'] += 1
    with pytest.raises(AssertionError):
        entrypoint.assert_restart(training, expected, 'cpu')

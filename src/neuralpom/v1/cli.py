"""Explicit campaign actions; importing this module never launches a run."""
from __future__ import annotations

import argparse
from datetime import timedelta
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist

from .checkpoint import distributed
from .config import load_config
from .execution import collective_call
from .runtime import Runtime
from .sampling import STAGES, stage_samples
from .tracking import ExperimentLogger


def initialize(config, requested_device=None):
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(requested_device or (f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if int(os.environ.get('WORLD_SIZE', 1)) > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo', timeout=timedelta(hours=2))
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return device, dist.get_rank() if distributed() else 0


def create_logger(config, stage, rank, resume=False):
    logger, error = None, None
    try:
        logger = ExperimentLogger(config, stage, rank=rank, resume=resume)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    errors = [error]
    if distributed():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
    if any(errors):
        if logger is not None:
            logger.finish(False)
        raise RuntimeError(f'Experiment tracking initialization failed: {errors}')
    return logger


def main(argv=None, action=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--device', default=None)
    if action is None:
        actions = parser.add_subparsers(dest='action', required=True)
        actions.add_parser('prepare', help='Build endpoint caches and training-only calibration')
        training = actions.add_parser('train')
    else:
        if action not in ('prepare', 'train'):
            raise ValueError('Use the evaluation entry point for frozen evaluation')
        parser.set_defaults(action=action)
        training = parser if action == 'train' else None
    if training is not None:
        training.add_argument('--stage', choices=STAGES, required=True)
        training.add_argument('--parent', type=Path)
        training.add_argument('--no-resume', action='store_true')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    logger = None
    success = False
    try:
        device, rank = initialize(config, args.device)
        stage = args.stage if args.action == 'train' else args.action
        logger = create_logger(config, stage, rank, resume=args.action == 'train' and not args.no_resume)
        runtime = collective_call(lambda: Runtime(config, device, rank, assets=args.action != 'prepare'))
        if args.action == 'prepare':
            from .cache import build_assets
            build_assets(config, runtime.core, runtime.reset,
                         lambda seconds: runtime.advance(False, seconds),
                         stage_samples(config, 'pretrain'),
                         stage_samples(config, 'pretrain', split='validation'), logger=logger)
        elif args.action == 'train':
            if args.stage == 'posttrain':
                from .posttrain import build_posttrain
                trainer = build_posttrain(config, runtime, logger, parent=args.parent,
                                          resume=not args.no_resume)
            else:
                from .engine import Training
                trainer = Training(config, args.stage, runtime, logger, parent=args.parent,
                                   resume=not args.no_resume)
            trainer.train()
        success = True
    finally:
        if logger is not None:
            logger.finish(success)
        if distributed():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()

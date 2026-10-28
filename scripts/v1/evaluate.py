"""Frozen model or pure-physics evaluation on registered held-out origins."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import torch.distributed as dist
from neuralpom.v1.checkpoint import distributed
from neuralpom.v1.cli import initialize, create_logger, collective_call
from neuralpom.v1.config import load_config
from neuralpom.v1.evaluation import evaluate
from neuralpom.v1.runtime import Runtime

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--weights', type=Path)
    mode.add_argument('--baseline', action='store_true')
    parser.add_argument('--days', type=int, required=True)
    parser.add_argument('--device')
    args = parser.parse_args()
    if args.days < 1:
        parser.error('--days must be positive')
    config = load_config(args.config)
    logger, success = None, False
    try:
        device, rank = initialize(config, args.device)
        label = 'physics' if args.baseline else args.weights.stem
        logger = create_logger(config, f'evaluation/{label}_{args.days}d', rank)
        runtime = collective_call(lambda: Runtime(config, device, rank))
        evaluate(config, runtime, args.weights, args.days, logger, baseline=args.baseline)
        success = True
    finally:
        if logger is not None:
            logger.finish(success)
        if distributed():
            dist.destroy_process_group()

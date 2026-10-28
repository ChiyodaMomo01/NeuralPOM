"""Render evaluation figures and optional paired skill comparison into W&B."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from neuralpom.v1.config import load_config
from neuralpom.v1.evaluation import compare_to_physics
from neuralpom.v1.visualization import render_evaluation, render_comparison
from neuralpom.v1.tracking import ExperimentLogger

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--physics', type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    logger = ExperimentLogger(config, f'reports/{args.evaluation.name}', resume=True)
    success = False
    try:
        for path in render_evaluation(args.evaluation):
            logger.image(path, f'figures/{path.stem}')
        if args.physics:
            path = compare_to_physics(args.evaluation, args.physics, args.evaluation / 'physics_comparison.json')
            logger.artifact(path, 'paired-physics-comparison')
            figure = render_comparison(path)
            logger.image(figure, 'figures/physics_rmse_ratio')
        success = True
    finally:
        logger.finish(success)

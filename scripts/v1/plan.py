"""Display registered sample counts and optimizer budgets without starting a run."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from neuralpom.v1.config import load_config
from neuralpom.v1.sampling import STAGES, stage_plan

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    print(json.dumps({'frequency': config['frequency'], 'core': config['core'],
                      'stages': [stage_plan(config, stage) for stage in STAGES]}, indent=2))

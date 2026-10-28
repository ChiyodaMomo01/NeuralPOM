"""Launch with torchrun --nproc_per_node=8 scripts/v1/train.py --config ... --stage ..."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from neuralpom.v1.cli import main

if __name__ == '__main__':
    main(action='train')

"""Plan, normalize, train, and evaluate daily full-state AI baselines."""
import argparse
import json
from .config import load_config, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "prepare", "train", "evaluate", "report"))
    parser.add_argument("--config")
    parser.add_argument("--stage", choices=("pretrain", "posttrain"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--days", type=int)
    parser.add_argument("--root")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.action == "report":
        if not args.root:
            parser.error("report requires --root")
        from .report import summarize
        print(json.dumps(summarize(args.root), indent=2)); return
    if not args.config:
        parser.error("--config is required")
    config = load_config(args.config)
    if args.action == "plan":
        print(json.dumps(plan(config), indent=2)); return
    if args.action in ("train", "evaluate") and args.stage is None:
        parser.error("--stage is required")
    if args.max_updates is not None and args.max_updates < 1:
        parser.error("--max-updates must be positive")
    from .distributed import setup, close
    rank, world, device = setup(config, args.cpu)
    try:
        if args.action == "prepare":
            from .prepare import prepare
            result = prepare(config, rank, world, device)
        elif args.action == "train":
            from .engine import train
            result = train(config, args.stage, rank, world, device, args.resume, args.max_updates)
        else:
            from .evaluation import evaluate
            result = evaluate(config, args.stage, rank, world, device, args.days)
        if rank == 0:
            print(json.dumps(result, indent=2))
    finally:
        close()


if __name__ == "__main__":
    main()

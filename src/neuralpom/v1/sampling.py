"""Model-time sample lists and complete-dataset epoch budgets."""
from __future__ import annotations

import math

STAGES = ("pretrain", "midtrain4", "posttrain")


def _stage(config, stage):
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage!r}")
    seconds = config.get("interval_seconds", 3600 if config["frequency"] == "1h" else 86400)
    if stage == "pretrain":
        cfg = config["stages"]["pretrain"]
        return cfg["max_epochs"], cfg["stride_days"], 1, seconds
    if stage.startswith("midtrain"):
        cfg = config["stages"]["midtrain"]
        steps = int(stage[-1])
        return cfg["epochs"], cfg["base_stride_days"] * steps, steps, seconds
    cfg = config["stages"]["posttrain"]
    steps, rest = divmod(cfg["rollout_days"] * 86400, seconds)
    if rest:
        raise ValueError("Rollout must contain a whole number of correction intervals")
    return cfg["epochs"], cfg["stride_days"], steps, seconds


def stage_samples(config, stage, split="train"):
    """Unique (member, model day) pairs, with every target inside the split."""
    _, stride, steps, seconds = _stage(config, stage)
    bounds = config["split"]
    if split not in ("train", "validation"):
        raise ValueError("stage_samples supports train or validation")
    if split == "validation":
        stride = bounds["validation_stride_days"]
    if stride < 1:
        raise ValueError("Sampling stride must be positive")
    end = bounds[f"{split}_end_day"]
    return [(member, day) for member in bounds[f"{split}_members"]
            for day in range(bounds[f"{split}_start_day"], end, stride)
            if day * 86400 + steps * seconds < end * 86400]


def post_windows(config):
    settings = config['stages']['posttrain']
    return settings['rollout_days'] * 24 // settings['tbptt_hours']


def stage_plan(config, stage):
    epochs, stride, steps, seconds = _stage(config, stage)
    count = len(stage_samples(config, stage))
    if count == 0:
        raise ValueError(f"No samples for {stage}")
    world_size = int(config["world_size"])
    batches = math.ceil(count / world_size)
    windows = post_windows(config) if stage == "posttrain" else 1
    updates = batches * windows
    return dict(stage=stage, epochs=epochs, stride_days=stride, rollout_steps=steps,
                interval_seconds=seconds,
                gradient_method=("last_correction_only" if config['frequency'] == '1h' else "tbptt") if stage == "posttrain" else "full_bptt" if stage.startswith("midtrain") else "offline_single_step",
                tbptt_seconds=config["stages"]["posttrain"]["tbptt_hours"] * 3600 if stage == "posttrain" else None, samples=count,
                trajectory_batches_per_epoch=batches, updates_per_epoch=updates,
                maximum_updates=updates * epochs,
                padded_samples_per_epoch=batches * world_size - count)

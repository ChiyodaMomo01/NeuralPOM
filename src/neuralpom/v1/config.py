"""Campaign configuration shared by both correction frequencies."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .core_spec import core_spec


DEFAULTS = {
    "core": "implicit_p0_fixed", "frequency": "1day", "seed": 0,
    "supervision": "post-correction-only-v1",
    "world_size": 8, "grid": {"im": 90, "jm": 82},
    "model": {"hidden_dim": 256},
    "split": {
        "train_members": list(range(7)), "train_start_day": 1000, "train_end_day": 7000,
        "validation_members": [7], "validation_start_day": 7000,
        "validation_end_day": 8000, "validation_stride_days": 60,
        "evaluation_members": [7], "evaluation_start_day": 8000,
        "evaluation_end_day": 9000,
        "evaluation_starts": [8000, 8064, 8128, 8192, 8256, 8320, 8384, 8448],
    },
    "stages": {
        "pretrain": {"max_epochs": 40, "patience": 5, "min_delta": 0.0, "stride_days": 6,
                     "checkpoint_every_updates": 100},
        "midtrain": {"epochs": 10, "horizons": [4], "base_stride_days": 6,
                     "checkpoint_every_updates": 10},
        "posttrain": {"epochs": 20, "rollout_days": 10, "tbptt_hours": 24, "stride_days": 60,
                      "checkpoint_every_updates": 1},
    },
    "optimizer": {
        "pretrain_lr": 5e-5, "pretrain_final_lr": 5e-6, "midtrain_lr": 3e-6,
        "posttrain_lr": 3e-6, "weight_decay": 1e-4, "clip_norm": 10.0,
        "warmup_epochs": 3, "posttrain_warmup_updates": 50,
    },
    "acceleration": {
        "mode": "eager", "cuda_graph": False, "checkpoint_physics": True,
        "chunk_steps": 6, "compiled_kernels": ["proft"],
    },
    "losses": [{"name": "weighted_mse", "weight": 1.0, "scale": 1.0}],
    "wandb": {"entity": None, "project": "neuralpom-v1", "mode": "online"},
    "evaluation": {"leads_days": [50, 200, 300]},
}


def _merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def resolve_config(value: dict, *, base_dir: str | Path = ".") -> dict:
    if "loss" in value:
        raise ValueError("V1 loss configuration uses 'losses'; legacy 'loss' is not supported")
    config = _merge(DEFAULTS, value)
    if config["supervision"] != "post-correction-only-v1":
        raise ValueError("Training supervision must use corrected fields only")
    spec = core_spec(config["core"])
    if config["frequency"] not in ("1h", "1day"):
        raise ValueError("frequency must be 1h or 1day")
    config["interval_seconds"] = 3600 if config["frequency"] == "1h" else 86400
    if int(config["world_size"]) < 1:
        raise ValueError("world_size must be positive")
    base = Path(base_dir).expanduser().resolve()
    for name in ("data_root", "run_root"):
        if not config.get(name):
            raise ValueError(f"Missing {name}")
        path = Path(config[name]).expanduser()
        config[name] = str(path.resolve() if path.is_absolute() else (base / path).resolve())
    for name, filename in (("input_stats", "input_stats.npz"), ("scales", "scales.json"), ("front_scales", "front_scales.json")):
        path = Path(config.get(name, Path(config["run_root"]) / "assets" / filename)).expanduser()
        config[name] = str(path.resolve() if path.is_absolute() else (base / path).resolve())
    split = config["split"]
    for name in ("train", "validation", "evaluation"):
        if not split[f"{name}_members"] or not all(isinstance(m, int) and 0 <= m < 8 for m in split[f"{name}_members"]):
            raise ValueError(f"Invalid {name} members")
        if len(set(split[f"{name}_members"])) != len(split[f"{name}_members"]):
            raise ValueError(f"Duplicate {name} members")
        if not 1 <= split[f"{name}_start_day"] < split[f"{name}_end_day"] <= 9000:
            raise ValueError(f"Invalid {name} model-time bounds")
    if set(split["train_members"]) & set(split["validation_members"] + split["evaluation_members"]):
        raise ValueError("Training and held-out members must be disjoint")
    if split["train_end_day"] > split["validation_start_day"] or split["validation_end_day"] > split["evaluation_start_day"]:
        raise ValueError("Train/validation/evaluation time ranges must not overlap")
    for stage in ("pretrain", "midtrain", "posttrain"):
        settings = config["stages"][stage]
        if int(settings.get("epochs", settings.get("max_epochs", 0))) <= 0:
            raise ValueError(f"Invalid epoch budget for {stage}")
        interval = settings["checkpoint_every_updates"]
        if type(interval) is not int or interval < 1:
            raise ValueError(f"checkpoint_every_updates must be a positive integer for {stage}")
    if config["stages"]["midtrain"]["horizons"] != [4]:
        raise ValueError("Midtrain uses a single 4-step rollout after Pretrain")
    post = config["stages"]["posttrain"]
    if post.get('method', 'truncated') not in ('truncated', 'rl'):
        raise ValueError('Posttrain method must be truncated or rl')
    if post.get('method') == 'rl':
        from .posttrain.policy import validate_settings
        if not isinstance(post.get('rl'), dict):
            raise ValueError('RL Posttrain requires a stages.posttrain.rl settings object')
        validate_settings(post['rl'], config['world_size'], post['rollout_days'])
    if config.get('cache_root'):
        path = Path(config['cache_root']).expanduser()
        config['cache_root'] = str(path.resolve() if path.is_absolute() else (base / path).resolve())
    hours, days = post["tbptt_hours"], post["rollout_days"]
    if type(hours) is not int or type(days) is not int or days <= 0 or hours <= 0 or hours % 24 or days * 24 % hours:
        raise ValueError("TBPTT windows must contain whole days and divide the rollout")
    mode = 'score_function' if post.get('method') == 'rl' else ('last_correction_only' if config['frequency'] == '1h' else 'full_window')
    if post.setdefault('gradient_mode', mode) != mode:
        raise ValueError(f"Post-train gradient_mode must be {mode} for {config['frequency']}")
    if config['frequency'] == '1h' and hours != 24:
        raise ValueError('Hourly Post-train updates once per day with only the final correction gradient')
    if config["acceleration"]["mode"] not in ("eager", "compiled_vertical", "strict_vertical", "strict_vertical_compiled"):
        raise ValueError("Unknown acceleration mode")
    kernels = config["acceleration"]["compiled_kernels"]
    if spec.name == 'explicit':
        if (config['acceleration']['mode'] != 'eager' or config['acceleration'].get('cg_backend', 'torch') != 'torch'
                or config['acceleration']['cuda_graph'] or kernels):
            raise ValueError('Explicit training requires eager physics, torch backend, and no implicit compiled kernels')
    elif not kernels or len(set(kernels)) != len(kernels) or any(k not in ("profq", "proft") for k in kernels):
        raise ValueError("compiled_kernels must select proft and/or profq without duplicates")
    if spec.name == 'implicit_p0_fixed' and config["acceleration"]["mode"].startswith("strict_vertical") and kernels != ["proft"]:
        raise ValueError("Strict vertical modes compile only proft; compiled_kernels must be ['proft']")
    if config["wandb"].get("mode") not in ("online", "offline"):
        raise ValueError("All experiments require W&B in online or explicit offline mode")
    if not config["wandb"].get("project"):
        raise ValueError("W&B project is required")
    return config


def load_config(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    return resolve_config(json.loads(path.read_text()), base_dir=path.parent)


def training_settings(config: dict) -> dict:
    """Readable restart settings; paths and tracking locations can be moved."""
    return {key: copy.deepcopy(config[key]) for key in (
        "core", "frequency", "seed", "world_size", "grid", "model", "split",
        "stages", "optimizer", "losses", "supervision",
    )}

"""Independent AI configuration, retaining the hybrid sampling protocol."""
from __future__ import annotations
import copy
import json
from pathlib import Path
from neuralpom.v1.config import DEFAULTS as HYBRID_DEFAULTS, _merge
from neuralpom.v1.core_spec import core_spec
from neuralpom.v1.sampling import stage_samples, stage_plan

DEFAULTS = {
    "schema": "neuralpom-ai-daily-v1", "core": "implicit_p0_fixed", "seed": 0,
    "world_size": 8, "frequency": "1day", "interval_seconds": 86400,
    "grid": {"im": 90, "jm": 82}, "split": copy.deepcopy(HYBRID_DEFAULTS["split"]),
    "stages": {k: copy.deepcopy(HYBRID_DEFAULTS["stages"][k]) for k in ("pretrain", "posttrain")},
    "optimizer": copy.deepcopy(HYBRID_DEFAULTS["optimizer"]),
    "model": {"architecture": "triton", "spatial_hidden_dim": 256,
              "temporal_hidden_dim": 512, "num_spatial_layers": 4, "num_temporal_layers": 8},
    "normalization": {"minimum_scale": 1e-6},
    "wandb": {"mode": "offline", "project": "neuralpom-ai-baselines"},
    "evaluation": {"days": 100, "starts": [8000 + 32*i for i in range(16)],
                   "member": 7, "reserved_days": [8800, 8900]},
}

DEFAULTS["stages"]["posttrain"].update(patience=5, min_delta=0.)

def resolve_config(value, base_dir="."):
    config = _merge(DEFAULTS, value)
    if config["schema"] != DEFAULTS["schema"]:
        raise ValueError("Unknown AI baseline schema")
    core_spec(config["core"])
    if config["frequency"] != "1day" or config["interval_seconds"] != 86400:
        raise ValueError("The AI baseline predicts exactly one day per network call")
    if set(config["stages"]) != {"pretrain", "posttrain"}:
        raise ValueError("AI training has exactly Pretrain and Posttrain; no Midtrain")
    post = config["stages"]["posttrain"]
    if post["rollout_days"] != 10 or post["tbptt_hours"] != 24:
        raise ValueError("AI Posttrain requires ten days and one-day parameter gradients")
    if config["model"]["architecture"] not in ("triton", "resunet", "simvp", "convlstm", "pastnet"):
        raise ValueError("Unknown AI architecture")
    if type(config["world_size"]) is not int or config["world_size"] < 1:
        raise ValueError("Invalid DDP world size")
    split = config["split"]
    for name in ("train", "validation", "evaluation"):
        members = split[f"{name}_members"]
        if not members or len(set(members)) != len(members) or any(type(m) is not int or not 0 <= m < 8 for m in members):
            raise ValueError("Invalid split members")
        if not 1 <= split[f"{name}_start_day"] < split[f"{name}_end_day"] <= 9000:
            raise ValueError("Invalid split bounds")
    if set(split["train_members"]) & set(split["validation_members"] + split["evaluation_members"]):
        raise ValueError("Training members must be disjoint from validation/evaluation")
    if split["train_end_day"] > split["validation_start_day"] or split["validation_end_day"] > split["evaluation_start_day"]:
        raise ValueError("Training, validation and evaluation time blocks overlap")
    for stage in ("pretrain", "posttrain"):
        settings = config["stages"][stage]
        for key in ("stride_days", "checkpoint_every_updates"):
            if type(settings[key]) is not int or settings[key] < 1:
                raise ValueError(f"Invalid {stage} {key}")
        if type(settings["patience"]) is not int or settings["patience"] < 1 or settings.get("min_delta", 0.) < 0:
            raise ValueError("Invalid validation early-stop settings")
        if settings.get("epochs", settings.get("max_epochs", 0)) <= 0:
            raise ValueError("Epoch budgets must be positive")
    if config["normalization"]["minimum_scale"] <= 0:
        raise ValueError("Normalization scale floor must be positive")
    for stage in ("pretrain", "posttrain"):
        if not stage_samples(config, stage) or not stage_samples(config, stage, "validation"):
            raise ValueError("AI training/validation sample partition is empty")
    base = Path(base_dir).expanduser().resolve()
    for key in ("data_root", "run_root"):
        if not config.get(key):
            raise ValueError(f"Missing {key}")
        path = Path(config[key]).expanduser()
        config[key] = str(path.resolve() if path.is_absolute() else (base/path).resolve())
    config["state_stats"] = str(Path(config["run_root"])/"assets/state_stats.npz")
    if config["wandb"]["mode"] not in ("online", "offline", "disabled"):
        raise ValueError("Invalid tracking mode")
    return config


def load_config(path):
    path = Path(path).resolve()
    return resolve_config(json.loads(path.read_text()), path.parent)


def plan(config):
    return {stage: dict(stage_plan(config, stage), gradient_method=(
        "one_day_tbptt" if stage == "posttrain" else "direct_next_day"))
        for stage in ("pretrain", "posttrain")}

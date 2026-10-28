"""AI trajectories in the common frozen campaign format and diagnostic pipeline."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import time
import traceback
import torch
from neuralpom.evaluation.protocol import atomic_json, case_root, read_json, sha256, FIELDS
from neuralpom.v1.checkpoint import atomic_torch
from neuralpom.v1.diagnostics import state_diagnostics
from .config import load_config
from .runtime import AIRuntime
from .models import build_model


def build_runtime(variant, device):
    from .distributed import configure_execution
    configure_execution()
    config = load_config(variant["config"])
    if config["core"] != variant["core"] or variant["frequency"] != "1day":
        raise ValueError("Registered AI reference core/frequency differs")
    runtime = AIRuntime(config, device)
    path = Path(config["run_root"])/"frozen"/f"{variant['stage']}.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["contract"] != runtime.contract() or checkpoint["stage"] != variant["stage"]:
        raise ValueError("AI frozen checkpoint contract differs")
    model = build_model(config["model"], runtime.codec).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().requires_grad_(False)
    metadata = {key: checkpoint.get(key) for key in ("step", "selection", "posttrain_gradient_mode")}
    metadata.update(training_contract=runtime.contract(), weights=str(path), weight_sha256=sha256(path),
        config_sha256=sha256(variant["config"]), assets_sha256={"state_stats": sha256(config["state_stats"])},
        inference={"model_family": "ai", "physical_forward_steps": 0, "reference_feedback": False,
                   "interval_seconds": 86400, "deterministic_algorithms": True, "tf32": False}, correction_policy="ai_based")
    return runtime, model, metadata


@torch.no_grad()
def run_case(protocol, variant, case, runtime, model, metadata, stop_after=None):
    from neuralpom.evaluation.runner import save_fields, read_records, append, verify_frozen
    output = case_root(protocol["output_root"], variant["id"], case)
    output.mkdir(parents=True, exist_ok=True)
    target = max(protocol["horizons_days"])
    horizon = target if stop_after is None else min(target, int(stop_after))
    identity = {"variant": variant["id"], **case, "target_days": target,
                "weight_sha256": metadata["weight_sha256"], "protocol_sha256": protocol["_sha256"],
                "state_stats_sha256": metadata["assets_sha256"]["state_stats"]}
    path = output/"status.json"
    if path.exists():
        old = read_json(path)
        if any(old.get(k) != v for k, v in identity.items()):
            raise ValueError("Refusing to mix different AI inference contracts")
        if old["status"] in ("complete", "numerical_failure") or old["completed_days"] >= horizon:
            return old
    initial = runtime.data.load(case["member"], case["start_day"], runtime.device)
    current = runtime.normalizer.encode(runtime.codec.pack(initial))
    frozen = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    geometry = {k: getattr(runtime.core, k) for k in ("fsm", "dum", "dvm", "h", "art", "dx", "dy", "dz", "zz")}
    save_fields(output/"geometry.npz", geometry, case["start_day"])
    save_fields(output/"initial.npz", {k: initial[k] for k in FIELDS}, case["start_day"])
    save_fields(output/"initial_fullstate.npz", {k: initial[k] for k in runtime.codec.contract()["fields"]}, case["start_day"])
    status = dict(identity, status="running", completed_days=0, pid=os.getpid(), updated_at=time.time(),
                  physical_forward_steps=0, reference_feedback=False)
    restart = output/"restart.pt"
    if restart.exists():
        saved = torch.load(restart, map_location="cpu", weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("AI restart identity differs")
        current = saved["encoded"].to(runtime.device)
        status["completed_days"] = saved["day"]
    for name in ("daily", "spectra", "profiles"):
        record_path = output/f"{name}.jsonl"
        rows = [row for row in read_records(record_path) if row["lead_day"] <= status["completed_days"]]
        if [row["lead_day"] for row in rows] != list(range(1, status["completed_days"]+1)):
            raise ValueError("Incomplete committed AI daily records")
        record_path.write_text("".join(json.dumps(row, allow_nan=False)+"\n" for row in rows))
    atomic_json(path, status)
    try:
        for day in range(status["completed_days"]+1, horizon+1):
            started = time.monotonic()
            # Generate the complete next state BEFORE reading the scoring target.
            current = runtime.predict(model, current).detach()
            prediction = runtime.diagnostic_state(current)
            reference = runtime.data.load(case["member"], case["start_day"]+day, runtime.device)
            metrics, profiles, spectra = state_diagnostics(runtime.core, prediction, reference)
            if not all(math.isfinite(value) for value in metrics.values()):
                raise FloatingPointError("Nonfinite AI diagnostic")
            clock = case["start_day"]+day
            save_fields(output/f"day{day:03d}_post.npz", {k: prediction[k] for k in FIELDS}, clock)
            save_fields(output/f"day{day:03d}_fullstate.npz", prediction, clock)
            save_fields(output/f"day{day:03d}_reference.npz", {k: reference[k] for k in FIELDS}, float(reference["time"]))
            append(output/"daily.jsonl", {"lead_day": day, "time": clock, "reference_time": float(reference["time"]),
                "metrics": metrics, "wall_seconds": time.monotonic()-started, "model_family": "ai",
                "correction_policy": "ai_based", "min_ts_limit_factor": 1.})
            append(output/"spectra.jsonl", dict(lead_day=day, **spectra))
            append(output/"profiles.jsonl", dict(lead_day=day, **profiles))
            atomic_torch(restart, {"day": day, "encoded": current.cpu(), "identity": identity})
            status.update(completed_days=day, updated_at=time.time())
            atomic_json(path, status)
        status["status"] = "complete" if horizon == target else "running"
    except FloatingPointError as error:
        status.update(status="numerical_failure", first_anomaly_day=status["completed_days"]+1,
                      first_anomaly_time=case["start_day"]+status["completed_days"]+1, error=str(error))
    except Exception as error:
        status.update(status="infrastructure_error", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        try:
            verify_frozen(model, frozen)
        finally:
            status["updated_at"] = time.time()
            atomic_json(path, status)
    return status


def evaluate(config, stage, rank, world, device, days=None):
    """Eight ranks partition origins; there is no DDP communication in forecasts."""
    import torch.distributed as dist
    from .distributed import phase
    from .report import summarize
    days = config["evaluation"]["days"] if days is None else days
    member = config["evaluation"]["member"]
    starts = config["evaluation"]["starts"]
    if type(days) is not int or days < 1 or not starts or len(set(starts)) != len(starts):
        raise ValueError("Invalid AI evaluation horizon/origins")
    reserved = config["evaluation"]["reserved_days"]
    lower, upper = config["split"]["evaluation_start_day"], config["split"]["evaluation_end_day"]
    if member not in config["split"]["evaluation_members"] or any(
        start < lower or start+days >= upper or any(start <= hold <= start+days for hold in reserved) for start in starts):
        raise ValueError("AI evaluation would leave its partition or touch a reserved holdout")
    root = Path(config["run_root"])/"evaluation"/f"{stage}_{days}d"
    variant = {"id": f"{config['core']}_ai_{config['model']['architecture']}_{stage}",
        "config": str(root/"config.json"), "core": config["core"], "frequency": "1day",
        "stage": stage, "policy": "ai_based", "model_family": "ai", "group_id": f"{config['core']}_1day"}
    def publish():
        root.mkdir(parents=True, exist_ok=True)
        if rank == 0:
            path = root/"config.json"
            if path.exists() and read_json(path) != config:
                raise ValueError("AI evaluation config changed; use a new output directory")
            atomic_json(path, config)
    phase(publish, device)
    runtime, model, metadata = phase(lambda: build_runtime(variant, device), device)
    cases = [{"member": member, "start_day": start} for start in starts]
    protocol = {"horizons_days": [days], "cases": cases, "output_root": str(root)}
    protocol["_sha256"] = sha256(root/"config.json")
    def manifest():
        if rank == 0:
            output = root/"runs"/variant["id"]
            output.mkdir(parents=True, exist_ok=True)
            value = dict(variant=variant, **metadata, protocol_sha256=protocol["_sha256"])
            path = output/"manifest.json"
            if path.exists() and read_json(path) != value:
                raise ValueError("AI evaluation manifest changed; use a new output directory")
            atomic_json(path, value)
            atomic_json(root/"manifest.json", {"variant": variant, "cases": cases, "days": days, **metadata})
    phase(manifest, device)
    for slot in range(0, len(cases), world):
        index = slot+rank
        phase(lambda: run_case(protocol, variant, cases[index], runtime, model, metadata) if index < len(cases) else None, device)
    phase(lambda: summarize(root) if rank == 0 else None, device)
    return str(root)

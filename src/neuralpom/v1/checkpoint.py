"""Complete physical and training state; atomic, per-rank restart generations."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist


SCRATCH = {"device", "_accel_cache", "_cg_work", "_eta_solver_iteration_history",
           "_spectral_eta_preconditioner_cache"}


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _scratch(name):
    return name in SCRATCH or name.startswith(("_compiled_", "_cuda_graph_"))


def capture(core, cpu=True):
    """Include every tensor, solver warm start, policy and scalar clock.

    Only explicitly identified computational workspaces are rebuilt. Unknown
    objects fail serialization instead of silently losing physical state.
    """
    tensors, scalars, objects = {}, {}, {}
    for name, value in vars(core).items():
        if _scratch(name):
            continue
        if torch.is_tensor(value):
            tensors[name] = value.detach().to("cpu" if cpu else value.device).clone()
        elif value is None or isinstance(value, (bool, int, float, str, np.generic)):
            scalars[name] = value.item() if isinstance(value, np.generic) else value
        elif name in {"_settings_overrides", "_implicit_forcing_provider",
                      "_implicit_numerics_policy", "_implicit_boundary_policy",
                      "_gradient_kernel_names"}:
            objects[name] = copy.deepcopy(value)
        else:
            raise TypeError(f"Unclassified physical state {name}: {type(value).__name__}")
    return {"tensors": tensors, "scalars": scalars, "objects": objects, "complete": True}


def restore(core, state):
    if state.get("complete") is not True:
        raise ValueError("Restart requires a complete physical state")
    device = core.device
    saved = set(state["tensors"]) | set(state["scalars"]) | set(state["objects"])
    for name in list(vars(core)):
        if not _scratch(name) and name not in saved:
            delattr(core, name)
    for name, value in state["tensors"].items():
        setattr(core, name, value.detach().to(device).clone())
    for section in ("scalars", "objects"):
        for name, value in state[section].items():
            setattr(core, name, copy.deepcopy(value))
    core.device = device
    core._cg_work = {}
    core._accel_cache = {}
    core._accel_cache_ready = False
    core._spectral_eta_preconditioner_cache = None
    for name in ("runtime_no_grad_fast_path", "runtime_cuda_graph_fast_path",
                 "runtime_vamax_safe_fast_path", "runtime_ramp_saturated_fast_path"):
        setattr(core, name, False)


def rng_state(device=None):
    device = torch.device(device) if device is not None else torch.device('cpu')
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}


def restore_rng(value, device=None):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"].cpu())
    if value["cuda"] is not None:
        torch.cuda.set_rng_state(value["cuda"].cpu(), device)


def distributed():
    return dist.is_available() and dist.is_initialized()


def unwrap(model):
    return model.module if hasattr(model, "module") else model


class RestartStore:
    """Publish a pointer only once every rank has saved its state."""

    def __init__(self, directory, contract):
        self.directory = Path(directory)
        self.contract = contract
        self.rank = dist.get_rank() if distributed() else 0
        self.world = dist.get_world_size() if distributed() else 1

    def _phase(self, action):
        result, error = None, None
        try:
            result = action()
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        failures = [error]
        if distributed():
            failures = [None] * self.world
            dist.all_gather_object(failures, error)
        if any(failures):
            raise RuntimeError(f'Checkpoint operation failed: {failures}')
        return result

    def save(self, model, optimizer, cursor, step, physical=None, extra=None):
        nonce = [str(time.time_ns()) if self.rank == 0 else None]
        if distributed():
            dist.broadcast_object_list(nonce, src=0)
        generation = self.directory / "restart" / f"step_{step:08d}_{nonce[0]}"
        local_path = generation / f"rank_{self.rank}.pt"
        device = next(unwrap(model).parameters()).device
        self._phase(lambda: atomic_torch(local_path, {"rank": self.rank, "step": step,
                                                     "rng": rng_state(device), "physical": physical}))
        info = {"path": str(local_path.relative_to(self.directory))}
        shards = [None] * self.world
        if distributed():
            dist.all_gather_object(shards, info)
        else:
            shards[0] = info
        def publish():
            if self.rank != 0:
                return
            training_path = generation / "training.pt"
            atomic_torch(training_path, {"model": cpu_tree(unwrap(model).state_dict()),
                         "optimizer": cpu_tree(optimizer.state_dict()), "contract": self.contract,
                         "cursor": cursor, "step": step, "world_size": self.world,
                         "rank_state": shards, "extra": extra or {}})
            atomic_json(self.directory / "resume.json", {
                "checkpoint": str(training_path.relative_to(self.directory))})
        self._phase(publish)
        def prune():
            if self.rank != 0:
                return
            # Only this store's generations, after the complete pointer is visible.
            generations = list((self.directory / "restart").glob("step_*"))
            previous = sorted((path for path in generations if (path / 'training.pt').is_file()),
                              key=lambda p: p.stat().st_mtime_ns)
            incomplete = [path for path in generations if not (path / 'training.pt').is_file()]
            for old in previous[:-2] + incomplete:
                if old != generation:
                    shutil.rmtree(old)
        self._phase(prune)

    def load(self, model, optimizer):
        return self._phase(lambda: self._load(model, optimizer))

    def _load(self, model, optimizer):
        pointer = json.loads((self.directory / "resume.json").read_text())
        path = self.directory / pointer["checkpoint"]
        value = torch.load(path, map_location="cpu", weights_only=False)
        if value["contract"] != self.contract or value["world_size"] != self.world:
            raise ValueError("Restart contract or world size mismatch")
        shard = value["rank_state"][self.rank]
        local_path = self.directory / shard["path"]
        local = torch.load(local_path, map_location="cpu", weights_only=False)
        if local["rank"] != self.rank or local["step"] != value["step"]:
            raise ValueError("Restart ranks differ")
        unwrap(model).load_state_dict(value["model"], strict=True)
        optimizer.load_state_dict(value["optimizer"])
        restore_rng(local["rng"], next(unwrap(model).parameters()).device)
        return value, local["physical"]

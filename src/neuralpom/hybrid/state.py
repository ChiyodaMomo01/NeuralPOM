from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from .constants import (
    INPUT_CHANNELS,
    INPUT_PACK_ORDER,
    INPUT_THREED_FIELDS,
    SUPERVISED_KB,
)


STATE_FALLBACKS = {
    "ub": "u",
    "vb": "v",
    "tb": "t",
    "sb": "s",
    "elb": "el",
    "uab": "ua",
    "vab": "va",
    "q2b": "q2",
    "q2lb": "q2l",
}

INJECTION_PAIRS = (
    ("u", "u"),
    ("ub", "ub"),
    ("v", "v"),
    ("vb", "vb"),
    ("t", "t"),
    ("tb", "tb"),
    ("s", "s"),
    ("sb", "sb"),
    ("el", "el"),
    ("elb", "elb"),
    ("ua", "ua"),
    ("uab", "uab"),
    ("va", "va"),
    ("vab", "vab"),
    ("q2", "q2"),
    ("q2b", "q2b"),
    ("q2l", "q2l"),
    ("q2lb", "q2lb"),
    ("km", "km"),
    ("kh", "kh"),
    ("kq", "kq"),
    ("rho", "rho"),
    ("w", "w"),
    ("et", "et"),
    ("etb", "etb"),
    ("etf", "etf"),
    ("utf", "utf"),
    ("utb", "utb"),
    ("vtf", "vtf"),
    ("vtb", "vtb"),
    ("adx2d", "adx2d"),
    ("ady2d", "ady2d"),
    ("aam", "aam"),
    ("vfluxf", "vfluxf"),
    ("vfluxb", "vfluxb"),
    ("wubot", "wubot"),
    ("wvbot", "wvbot"),
    ("egb", "egb"),
    ("drx2d", "drx2d"),
    ("dry2d", "dry2d"),
    ("uaf", "uaf"),
    ("vaf", "vaf"),
    ("elf", "elf"),
)

OPTIONAL_ZERO_FIELDS = {"drx2d", "dry2d"}


@dataclass
class StateNormalizer:
    means: dict[str, torch.Tensor]
    scales: dict[str, torch.Tensor]
    format_version: int
    source_path: str
    metadata: dict[str, object]

    @property
    def t_mean(self) -> torch.Tensor:
        return self.means["t"]

    @property
    def s_mean(self) -> torch.Tensor:
        return self.means["s"]

    @classmethod
    def from_path(cls, path: str | Path, device: torch.device) -> "StateNormalizer":
        mean_path = Path(path)
        if not mean_path.exists():
            raise FileNotFoundError(
                f"Input-normalization file not found: {mean_path}. "
                "Online training must never fall back to unregistered statistics."
            )
        with np.load(mean_path, allow_pickle=False) as mean_data:
            files = set(mean_data.files)
            version = int(np.asarray(mean_data["format_version"]).item()) if "format_version" in files else 1
            means: dict[str, torch.Tensor] = {}
            scales: dict[str, torch.Tensor] = {}

            if version >= 2:
                missing = {
                    key
                    for field in INPUT_PACK_ORDER
                    for key in (f"mean__{field}", f"scale__{field}")
                    if key not in files
                }
                if missing:
                    raise KeyError(
                        f"Input-normalization file {mean_path} is missing registered keys: {sorted(missing)}"
                    )
                for field in INPUT_PACK_ORDER:
                    mean_array = np.asarray(mean_data[f"mean__{field}"], dtype=np.float32)
                    scale_array = np.asarray(mean_data[f"scale__{field}"], dtype=np.float32)
                    if mean_array.shape != scale_array.shape:
                        raise ValueError(
                            f"Normalizer shape mismatch for {field}: {mean_array.shape} vs {scale_array.shape}"
                        )
                    if not np.isfinite(mean_array).all() or not np.isfinite(scale_array).all():
                        raise ValueError(f"Normalizer contains non-finite values for field {field}")
                    if not np.all(scale_array > 0.0):
                        raise ValueError(f"Normalizer scale must be strictly positive for field {field}")
                    means[field] = torch.from_numpy(mean_array).to(device=device).unsqueeze(0)
                    scales[field] = torch.from_numpy(scale_array).to(device=device).unsqueeze(0)
                metadata = (
                    json.loads(str(np.asarray(mean_data["metadata_json"]).item()))
                    if "metadata_json" in files
                    else {}
                )
            else:
                missing_keys = {"t", "s"} - files
                if missing_keys:
                    raise KeyError(f"Legacy mean-state file {mean_path} is missing keys: {sorted(missing_keys)}")
                for field in ("t", "s"):
                    value = torch.from_numpy(np.asarray(mean_data[field], dtype=np.float32)).to(device=device).unsqueeze(0)
                    means[field] = value
                    scales[field] = torch.ones_like(value)
                metadata = {"legacy": True}

        return cls(
            means=means,
            scales=scales,
            format_version=version,
            source_path=str(mean_path.resolve()),
            metadata=metadata,
        )

    def normalize(self, tensor: torch.Tensor, var_name: str) -> torch.Tensor:
        if var_name not in self.means:
            return tensor
        mean = self.means[var_name]
        scale = self.scales[var_name]
        if tensor.dim() == mean.dim() - 1:
            mean = mean[0]
            scale = scale[0]
        elif tensor.dim() == mean.dim() and mean.shape[0] == 1:
            # Registered statistics carry a singleton batch axis. Keep that
            # axis broadcastable for diagnostics and future larger batches.
            if tensor.shape[1:] != mean.shape[1:]:
                raise ValueError(
                    f"Normalizer shape mismatch for {var_name}: tensor={tuple(tensor.shape)}, "
                    f"registered={tuple(mean.shape)}"
                )
        elif tensor.shape != mean.shape:
            raise ValueError(
                f"Normalizer shape mismatch for {var_name}: tensor={tuple(tensor.shape)}, "
                f"registered={tuple(mean.shape)}"
            )
        return (tensor - mean) / scale

    def denormalize(self, tensor: torch.Tensor, var_name: str) -> torch.Tensor:
        return tensor


def _ensure_channels_first(tensor: torch.Tensor, field_name: str) -> torch.Tensor:
    if field_name in INPUT_THREED_FIELDS:
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() == 4:
            tensor = tensor.permute(0, 3, 1, 2)
        return tensor
    if tensor.dim() == 2:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.dim() == 3:
        return tensor.unsqueeze(1)
    return tensor


def pack_state_tensors(state: dict[str, torch.Tensor], normalizer: StateNormalizer) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    for field_name in INPUT_PACK_ORDER:
        tensor = normalizer.normalize(state[field_name], field_name)
        pieces.append(_ensure_channels_first(tensor, field_name))
    return torch.cat(pieces, dim=1)


def collect_pom_batch_state(poms: list, kb: int = SUPERVISED_KB) -> dict[str, torch.Tensor]:
    state = {key: [] for key in INPUT_PACK_ORDER}
    for pom in poms:
        state["u"].append(pom.u[:, :, :kb])
        state["ub"].append(pom.ub[:, :, :kb])
        state["v"].append(pom.v[:, :, :kb])
        state["vb"].append(pom.vb[:, :, :kb])
        state["t"].append(pom.t[:, :, :kb])
        state["s"].append(pom.s[:, :, :kb])
        state["el"].append(pom.el)
        state["ua"].append(pom.ua)
        state["uab"].append(pom.uab)
        state["va"].append(pom.va)
        state["vab"].append(pom.vab)
    return {key: torch.stack(values) for key, values in state.items()}


def pack_pom_batch(poms: list, normalizer: StateNormalizer, kb: int = SUPERVISED_KB) -> torch.Tensor:
    return pack_state_tensors(collect_pom_batch_state(poms, kb=kb), normalizer)




def inject_core_state(pom, ics: dict[str, torch.Tensor], sample_index: int | None = None, start_day: int | None = None) -> None:
    def resolve_value(source_key: str) -> torch.Tensor | None:
        if source_key in ics:
            value = ics[source_key]
        elif source_key in STATE_FALLBACKS and STATE_FALLBACKS[source_key] in ics:
            value = ics[STATE_FALLBACKS[source_key]]
        else:
            return None

        if sample_index is not None and hasattr(value, "dim") and hasattr(pom, source_key if hasattr(pom, source_key) else next(iter(STATE_FALLBACKS), source_key)):
            target_attr_name = source_key
            if not hasattr(pom, target_attr_name) and source_key in STATE_FALLBACKS:
                target_attr_name = STATE_FALLBACKS[source_key]
            target = getattr(pom, target_attr_name, None)
            if target is not None and value.dim() > target.dim():
                value = value[sample_index]
        return value

    if hasattr(pom, "_eta_prev_solution"):
        pom._eta_prev_solution = None

    for attr_name, source_key in INJECTION_PAIRS:
        target_attr = getattr(pom, attr_name, None)
        if target_attr is None:
            continue

        value = resolve_value(source_key)
        if value is None:
            if attr_name in OPTIONAL_ZERO_FIELDS:
                setattr(pom, attr_name, torch.zeros_like(target_attr))
            continue

        value = value.detach().clone().to(pom.device)
        if value.dim() > target_attr.dim() and value.shape[0] == 1:
            value = value.squeeze(0)
        if value.shape != target_attr.shape:
            if value.numel() != target_attr.numel():
                continue
            value = value.view_as(target_attr)
        if target_attr.dtype == value.dtype and target_attr.device == value.device and target_attr.shape == value.shape:
            target_attr.detach_()
            target_attr.copy_(value)
        else:
            setattr(pom, attr_name, value)

    # The checkpoint clock is authoritative and is stored in days, matching
    # both explicit and implicit POM time loops.  Falling back to start_day is
    # retained for legacy callers that construct states without a checkpoint,
    # but start_day is also already measured in days (never seconds).
    clock_value = ics.get("time")
    if clock_value is not None:
        if isinstance(clock_value, torch.Tensor):
            clock_value = clock_value.detach()
            if sample_index is not None and clock_value.dim() > 0:
                if sample_index >= clock_value.shape[0]:
                    raise IndexError(
                        f"sample_index={sample_index} is out of range for checkpoint "
                        f"clock with shape {tuple(clock_value.shape)}"
                    )
                clock_value = clock_value[sample_index]
            if clock_value.numel() != 1:
                raise ValueError(
                    "Checkpoint clock must contain exactly one value per sample, "
                    f"got shape {tuple(clock_value.shape)}"
                )
            current_time_days = float(clock_value.item())
        else:
            current_time_days = float(clock_value)
    elif start_day is not None:
        current_time_days = float(start_day)
    else:
        current_time_days = None

    if current_time_days is not None:
        if not torch.isfinite(torch.tensor(current_time_days)):
            raise ValueError(f"Checkpoint clock must be finite, got {current_time_days}")
        pom.time = current_time_days
        pom.time0 = current_time_days
        pom.iint = 0

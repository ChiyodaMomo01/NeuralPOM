"""Lossless packing and frozen per-field/per-level full-state normalization."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from neuralpom.v1.data import INITIAL_3D_FIELDS, INITIAL_2D_FIELDS

FIELDS_3D = INITIAL_3D_FIELDS
FIELDS_2D = INITIAL_2D_FIELDS
FIELDS = (*FIELDS_3D, *FIELDS_2D)
U_FIELDS = {"u", "ub", "ua", "uab", "uaf", "utf", "utb", "adx2d", "drx2d", "wubot"}
V_FIELDS = {"v", "vb", "va", "vab", "vaf", "vtf", "vtb", "ady2d", "dry2d", "wvbot"}


class StateCodec:
    """All 43 spatial checkpoint fields; clock and counter remain metadata."""
    def __init__(self, grid=(90, 82, 16)):
        self.grid = tuple(grid)
        self.layout = [(name, self.grid[2] if name in FIELDS_3D else 1) for name in FIELDS]
        self.channels = sum(count for _, count in self.layout)

    def pack(self, state):
        pieces = []
        missing = set(FIELDS) - set(state)
        if missing:
            raise ValueError(f"Incomplete AI physical state: {sorted(missing)}")
        for name, count in self.layout:
            value = torch.as_tensor(state[name])
            shape = self.grid if name in FIELDS_3D else self.grid[:2]
            if tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
                raise ValueError(f"Invalid state field {name}: expected {shape}")
            pieces.append(value.permute(2, 0, 1) if name in FIELDS_3D else value.unsqueeze(0))
        return torch.cat(pieces, dim=0)

    def unpack(self, tensor):
        if tensor.shape != (self.channels, *self.grid[:2]):
            raise ValueError("AI output does not match the complete-state channel contract")
        result, offset = {}, 0
        for name, count in self.layout:
            value = tensor[offset:offset+count]
            result[name] = value.permute(1, 2, 0) if name in FIELDS_3D else value[0]
            offset += count
        return result

    def contract(self):
        return {"fields": list(FIELDS), "layout": self.layout, "grid": self.grid,
                "channels": self.channels, "storage_levels": self.grid[2],
                "clock": "advance one day deterministically; never supplied from future reference"}

    def support(self, core):
        pieces = []
        for name, count in self.layout:
            mask = core.dum if name in U_FIELDS else core.dvm if name in V_FIELDS else core.fsm
            pieces.append(mask.unsqueeze(0).expand(count, -1, -1) > .5)
        return torch.cat(pieces)

    def weights(self, core, reference=None):
        """One spatial reduction per channel, including every storage plane."""
        depth = core.h + (reference["et"] if reference is not None else core.et)
        if not bool((depth[core.fsm > .5] > 0).all()):
            raise ValueError("Nonpositive wet reference depth")
        pieces = []
        for name, count in self.layout:
            mask = core.dum if name in U_FIELDS else core.dvm if name in V_FIELDS else core.fsm
            weight = core.art * mask
            if name in FIELDS_3D:
                weight = weight * depth
            pieces.append(weight.unsqueeze(0).expand(count, -1, -1))
        return torch.cat(pieces)


class StateNormalizer:
    def __init__(self, mean, scale, metadata=None):
        self.mean = torch.as_tensor(mean).float().reshape(-1, 1, 1)
        self.scale = torch.as_tensor(scale).float().reshape(-1, 1, 1)
        if self.mean.shape != self.scale.shape or not bool(torch.isfinite(self.mean).all()) or not bool(torch.isfinite(self.scale).all()) or not bool((self.scale > 0).all()):
            raise ValueError("Full-state normalization requires finite means and positive scales")
        self.metadata = metadata or {}

    def to(self, device):
        self.mean, self.scale = self.mean.to(device), self.scale.to(device)
        return self

    def encode(self, value):
        return (value-self.mean)/self.scale

    def decode(self, value):
        return value*self.scale+self.mean

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez(temporary, mean=self.mean.cpu().numpy().ravel(), scale=self.scale.cpu().numpy().ravel(),
                 metadata=np.asarray(json.dumps(self.metadata, sort_keys=True)))
        temporary.replace(path)

    @classmethod
    def load(cls, path, device="cpu"):
        with np.load(path, allow_pickle=False) as values:
            result = cls(values["mean"], values["scale"], json.loads(str(values["metadata"])))
        return result.to(device)


def standardized_loss(prediction, target, weights):
    """Equal normalized-channel weights; reference geometry is shared spatially."""
    if prediction.shape != target.shape or prediction.shape != weights.shape:
        raise ValueError("Loss state/target/geometry shapes differ")
    if not bool(torch.isfinite(prediction).all()):
        raise FloatingPointError("Nonfinite AI prediction")
    sums = weights.sum((-2, -1))
    if not bool((sums > 0).all()):
        raise ValueError("AI loss channel has no wet support")
    return (((prediction-target).square()*weights).sum((-2, -1))/sums).mean()

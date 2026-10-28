"""Train-only complete-state moments, sharded across all DDP ranks."""
from __future__ import annotations
import json
from pathlib import Path
import torch
import torch.distributed as dist
from neuralpom.v1.sampling import stage_samples
from neuralpom.evaluation.protocol import atomic_json
from .runtime import AIRuntime
from .state import StateNormalizer
from .distributed import phase


def prepare(config, rank, world, device):
    runtime = phase(lambda: AIRuntime(config, device, assets=False), device)
    samples = stage_samples(config, "pretrain")
    sums = torch.zeros(3, runtime.codec.channels, dtype=torch.float64, device=device)
    # Same number of collective slots even when the last shard is incomplete.
    for start in range(0, len(samples), world):
        index = start+rank
        def accumulate():
            if index >= len(samples):
                return
            member, day = samples[index]
            state = runtime.data.load(member, day, device)
            x = runtime.codec.pack(state).double()
            w = runtime.codec.weights(runtime.core, state).double()
            sums[0].add_((x*w).sum((-2, -1)))
            sums[1].add_((x.square()*w).sum((-2, -1)))
            sums[2].add_(w.sum((-2, -1)))
        phase(accumulate, device)
    if dist.is_initialized():
        dist.all_reduce(sums)
    mean = sums[0]/sums[2]
    scale = (sums[1]/sums[2]-mean.square()).clamp_min(0).sqrt().clamp_min(config["normalization"]["minimum_scale"])
    metadata = {"data_identity": runtime.data.reference.data_identity, "split": config["split"],
                "codec": json.loads(json.dumps(runtime.codec.contract())), "stage": "pretrain",
                "sample_count": len(samples), "samples": samples,
                "normalization_source": "training inputs only; no validation/evaluation labels",
                "sampling": "same member/day/stride and end-boundary rule as hybrid daily Pretrain"}
    def publish():
        if rank != 0:
            return
        path = Path(config["state_stats"])
        if path.exists():
            raise FileExistsError("AI normalization already exists; use a new run directory")
        StateNormalizer(mean.cpu(), scale.cpu(), metadata).save(path)
        atomic_json(path.with_suffix(".json"), metadata)
    phase(publish, device)
    return metadata

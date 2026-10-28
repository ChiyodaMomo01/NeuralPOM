"""Geometry and metrics reuse without entering a physical integration loop."""
from __future__ import annotations
import json
import torch
from neuralpom.hybrid import build_initialized_core
from neuralpom.v1.core_spec import core_spec
from neuralpom.evaluation.protocol import sha256
from .data import DailyStateDataset
from .state import StateNormalizer


class AIRuntime:
    def __init__(self, config, device="cpu", assets=True):
        self.config, self.device = config, torch.device(device)
        self.data = DailyStateDataset(config)
        self.codec = self.data.codec
        spec = core_spec(config["core"])
        # Native geometry initialization contains CUDA cumsum, which this
        # PyTorch version does not classify as deterministic. It is outside
        # the neural forecast; restore strict neural arithmetic immediately.
        deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(False)
            self.core = build_initialized_core(spec.core_class, self.device, *self.codec.grid[:2])
        finally:
            torch.use_deterministic_algorithms(deterministic)
        spec.configure(self.core)
        self.normalizer = StateNormalizer.load(config["state_stats"], self.device) if assets else None
        self.physical_forward_steps = 0
        if assets:
            metadata = self.normalizer.metadata
            if metadata["data_identity"] != self.data.reference.data_identity or metadata["split"] != config["split"] or metadata["codec"] != json.loads(json.dumps(self.codec.contract())):
                raise ValueError("AI normalization, schema, split and dataset identities differ")

    def encoded(self, member, day):
        return self.normalizer.encode(self.codec.pack(self.data.load(member, day, self.device)))

    def predict(self, model, current):
        """One neural day; dry storage stays at its own initial boundary value."""
        proposed = model(current.unsqueeze(0))[0]
        if proposed.shape != current.shape or not bool(torch.isfinite(proposed).all()):
            raise FloatingPointError("Invalid complete AI next-state output")
        wet = self.codec.support(self.core)
        return torch.where(wet, proposed, current)

    def contract(self):
        return {"schema": "neuralpom-ai-daily-v1", "reference_core": self.config["core"],
                "data_identity": self.data.reference.data_identity, "split": self.config["split"],
                "codec": json.loads(json.dumps(self.codec.contract())), "interval_seconds": 86400,
                "model": self.config["model"], "state_stats_sha256": sha256(self.config["state_stats"]),
                "prediction": "direct normalized next-day complete state", "physical_forward_steps": 0}

    def diagnostic_state(self, packed):
        state = self.codec.unpack(self.normalizer.decode(packed))
        for key, value in state.items():
            setattr(self.core, key, value)
        return state

"""The same registered member/day loader, loading complete daily targets."""
from neuralpom.v1.data import FixedCoreDataset
from neuralpom.v1.sampling import stage_samples
from .state import StateCodec


class DailyStateDataset:
    def __init__(self, config):
        self.reference = FixedCoreDataset(config["data_root"], 86400, config["core"])
        expected = (config["grid"]["im"], config["grid"]["jm"], 16)
        if self.reference.grid != expected:
            raise ValueError("AI grid and registered reference grid differ")
        self.codec = StateCodec(self.reference.grid)

    def load(self, member, day, device="cpu"):
        return self.reference.load_initial(member, day, device)

    def pairs(self, config, stage, split="train"):
        return stage_samples(config, stage, split)

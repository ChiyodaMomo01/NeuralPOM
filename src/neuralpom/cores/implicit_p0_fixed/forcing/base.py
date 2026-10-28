from __future__ import annotations

from typing import Any, Protocol


class ForcingProvider(Protocol):
    name: str

    def initialize(self, core: Any) -> None: ...
    def update_before_step(self, core: Any, time_days: float, *, generation_fast_path: bool = False) -> None: ...

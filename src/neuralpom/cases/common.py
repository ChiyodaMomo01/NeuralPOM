from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class CaseSettings:
    name: str
    problem_id: int
    im: int
    jm: int
    description: str
    overrides: dict[str, Any] = field(default_factory=dict)


class CaseSetup(Protocol):
    def __call__(self, core: Any) -> None: ...


@dataclass(frozen=True)
class CaseSpec:
    name: str
    family: str
    im: int
    jm: int
    description: str
    setup_method: str | None = None
    boundary: str = "open"
    forcing: str = "none"
    numerics: str = "legacy"
    diagnostics: str | None = None
    legacy_problem_id: int | None = None
    overrides: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_settings(
        cls,
        settings: CaseSettings,
        *,
        family: str,
        setup_method: str | None = None,
        boundary: str = "open",
        forcing: str = "none",
        numerics: str = "legacy",
        diagnostics: str | None = None,
    ) -> "CaseSpec":
        return cls(
            name=settings.name,
            family=family,
            im=settings.im,
            jm=settings.jm,
            description=settings.description,
            setup_method=setup_method,
            boundary=boundary,
            forcing=forcing,
            numerics=numerics,
            diagnostics=diagnostics,
            legacy_problem_id=settings.problem_id,
            overrides=dict(settings.overrides),
        )

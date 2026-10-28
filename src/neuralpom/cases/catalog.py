from __future__ import annotations

from neuralpom.cases.common import CaseSettings, CaseSpec


DOUBLE_GYRE_SETTINGS = CaseSettings(
    name="double_gyre",
    problem_id=4,
    im=90,
    jm=82,
    description="Wind-driven double gyre used by NeuralPOM training and inference.",
)

CASE_SETTINGS_REGISTRY: dict[str, CaseSettings] = {
    DOUBLE_GYRE_SETTINGS.name: DOUBLE_GYRE_SETTINGS,
}

CASE_SPEC_REGISTRY: dict[str, CaseSpec] = {
    DOUBLE_GYRE_SETTINGS.name: CaseSpec.from_settings(
        DOUBLE_GYRE_SETTINGS,
        family="implicit_p0_fixed",
        setup_method="double_gyre",
        boundary="closed_double_gyre",
        forcing="double_gyre_wind",
    ),
}


def get_case_settings(name: str) -> CaseSettings:
    try:
        return CASE_SETTINGS_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown case {name!r}. Available: {sorted(CASE_SETTINGS_REGISTRY)}"
        ) from exc


def resolve_case_spec(name: str) -> CaseSpec:
    try:
        return CASE_SPEC_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown case spec {name!r}. Available: {sorted(CASE_SPEC_REGISTRY)}"
        ) from exc

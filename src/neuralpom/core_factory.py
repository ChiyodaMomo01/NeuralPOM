"""Production dynamical cores used by the common training pipeline."""
from neuralpom.cores.explicit import POM2K_Core as ExplicitCore
from neuralpom.cores.implicit_p0_fixed import POM2K_Core as ImplicitCore

CORE_REGISTRY = {"implicit_p0_fixed": ImplicitCore, "explicit": ExplicitCore}


def resolve_core_class(name: str):
    if name not in CORE_REGISTRY:
        raise ValueError(
            f"Unsupported production core {name!r}; choose from {sorted(CORE_REGISTRY)}."
        )
    return CORE_REGISTRY[name]

from __future__ import annotations

from .base import BaseCore
from .grid import GridMixin
from .cases import CasesMixin
from .case_policy import CasePolicyMixin
from .initialization import InitializationMixin
from .implicit_barotropic import ImplicitBarotropicMixin
from .baroclinic import BaroclinicMixin
from .tracers import TracersMixin
from .momentum import MomentumMixin
from .diagnostics import DiagnosticsMixin
from .checkpointing import CheckpointingMixin
from .time_loop import TimeLoopMixin
from .reference_base import ReferenceCoreBase


class POM2K_Core(
    GridMixin,
    CasePolicyMixin,
    CasesMixin,
    InitializationMixin,
    ImplicitBarotropicMixin,
    BaroclinicMixin,
    TracersMixin,
    MomentumMixin,
    DiagnosticsMixin,
    CheckpointingMixin,
    TimeLoopMixin,
    ReferenceCoreBase,
    BaseCore,
):
    pass

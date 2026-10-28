from __future__ import annotations

from typing import Any

from neuralpom.cases.catalog import resolve_case_spec
from neuralpom.cases.common import CaseSettings


class ReferenceCoreBase:
    def __init__(
        self,
        im: int,
        jm: int,
        problem_id: int = 4,
        settings_overrides: dict[str, Any] | None = None,
        case_name: str | None = None,
        case_spec=None,
    ):
        self._requested_problem_id = problem_id
        self._settings_overrides = dict(settings_overrides or {})
        self.case_name = case_name
        self.case_spec = case_spec
        super().__init__(im, jm)
        self._apply_settings_overrides()

    @classmethod
    def from_settings(cls, settings: CaseSettings):
        return cls(
            im=settings.im,
            jm=settings.jm,
            problem_id=settings.problem_id,
            settings_overrides=settings.overrides,
            case_name=settings.name,
            case_spec=resolve_case_spec(settings.name),
        )

    def _apply_settings_overrides(self) -> None:
        if not self._settings_overrides:
            return
        for key, value in self._settings_overrides.items():
            setattr(self, key, value)
        self._refresh_derived_indices()
        self._refresh_derived_time_params()

    def _refresh_derived_indices(self) -> None:
        self.imm1 = self.im - 1
        self.imm2 = self.im - 2
        self.jmm1 = self.jm - 1
        self.jmm2 = self.jm - 2
        self.kbm1 = self.kb - 1
        self.kbm2 = self.kb - 2

    def _refresh_derived_time_params(self) -> None:
        self.dti = self.dte * self.isplit
        self.dte2 = self.dte * 2.0
        self.dti2 = self.dti * 2.0
        self.iend = max(int((self.days * 24.0 * 3600.0 / self.dti) + 0.5), 2)
        self.iprint = int((self.prtd1 * 24.0 * 3600.0 / self.dti) + 0.5)
        self.iswtch = int((self.swtch * 24.0 * 3600.0 / self.dti) + 0.5)
        self.ispi = 1.0 / self.isplit
        self.isp2i = 1.0 / (2.0 * self.isplit)

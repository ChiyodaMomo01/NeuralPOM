from __future__ import annotations

from dataclasses import dataclass

from neuralpom.cores.implicit_p0_fixed.boundary import ClosedBasinBoundary
from neuralpom.cores.implicit_p0_fixed.forcing import DoubleGyreWindForcing
from neuralpom.cores.implicit_p0_fixed.numerics import DoubleGyreNumerics


@dataclass(frozen=True)
class ImplicitCasePolicy:
    name: str = "double_gyre"
    setup_method: str = "double_gyre"
    defines_vertical_grid: bool = True
    boundary_mode: str = "closed_double_gyre"
    surface_forcing: str = "double_gyre_wind"
    numerics: str = "double_gyre"


DOUBLE_GYRE_POLICY = ImplicitCasePolicy()
IMPLICIT_CASE_POLICIES = {4: DOUBLE_GYRE_POLICY}


class CasePolicyMixin:
    def implicit_case_policy(self) -> ImplicitCasePolicy:
        if int(self.iproblem) != 4:
            raise ValueError(
                f"Only the Double-gyre problem_id=4 is supported; got {self.iproblem}."
            )
        case_spec = getattr(self, "case_spec", None)
        if case_spec is not None and case_spec.name != "double_gyre":
            raise ValueError(f"Only the Double-gyre case is supported; got {case_spec.name!r}.")
        return DOUBLE_GYRE_POLICY

    def implicit_boundary_mode(self) -> str:
        return DOUBLE_GYRE_POLICY.boundary_mode

    def implicit_boundary_policy(self):
        policy = getattr(self, "_implicit_boundary_policy", None)
        if policy is None:
            policy = ClosedBasinBoundary()
            self._implicit_boundary_policy = policy
        return policy

    def implicit_forcing_provider(self):
        provider = getattr(self, "_implicit_forcing_provider", None)
        if provider is None:
            provider = DoubleGyreWindForcing()
            self._implicit_forcing_provider = provider
        return provider

    def implicit_numerics_policy(self):
        policy = getattr(self, "_implicit_numerics_policy", None)
        if policy is None:
            policy = DoubleGyreNumerics()
            self._implicit_numerics_policy = policy
        return policy

    def implicit_case_defines_vertical_grid(self) -> bool:
        return True

    def run_implicit_case_setup(self) -> None:
        self.double_gyre()
        self.implicit_forcing_provider().initialize(self)
        self.implicit_numerics_policy().initialize(self)

    def apply_implicit_case_surface_forcing(self, *, generation_fast_path: bool = False) -> None:
        self.implicit_forcing_provider().update_before_step(
            self,
            float(getattr(self, "time", 0.0)),
            generation_fast_path=generation_fast_path,
        )

    def apply_implicit_boundary_scalar2d(self, field):
        return self.implicit_boundary_policy().apply_scalar2d(self, field)

    def apply_implicit_boundary_velocity2d(self, u_field, v_field):
        return self.implicit_boundary_policy().apply_velocity2d(self, u_field, v_field)

    def apply_implicit_boundary_scalar3d(self, field):
        return self.implicit_boundary_policy().apply_scalar3d(self, field)

    def apply_implicit_boundary_velocity3d(self, u_field, v_field):
        return self.implicit_boundary_policy().apply_velocity3d(self, u_field, v_field)

    def apply_implicit_tracer_boundary(self) -> None:
        self.elf, self.uaf, self.vaf, self.t, self.s, self.w = (
            self.implicit_boundary_policy().apply_tracers(self)
        )

    def apply_implicit_flux2d_boundary(self, xflux, yflux):
        return self.implicit_boundary_policy().apply_flux2d(self, xflux, yflux)

    def apply_implicit_flux3d_boundary(self, xflux, yflux, k_slice):
        return self.implicit_boundary_policy().apply_flux3d(self, xflux, yflux, k_slice)

    def apply_implicit_bcond(self, idx, elf, uaf, vaf, uf, vf, w):
        return self.implicit_boundary_policy().apply_bcond(
            self, idx, elf, uaf, vaf, uf, vf, w
        )

    def implicit_boundary_refreshes_tracer_backups(self) -> bool:
        return self.implicit_boundary_policy().refresh_tracer_backups_after_apply()

    def enforce_implicit_eta_boundary(self, eta) -> None:
        self.implicit_boundary_policy().enforce_eta(self, eta)

    def prepare_implicit_eta_rhs(self, rhs) -> None:
        self.implicit_boundary_policy().prepare_eta_rhs(self, rhs)

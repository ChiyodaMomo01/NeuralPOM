"""Small-grid tests of the real physical implementation (not a mock core)."""
import io
import unittest

import torch

from neuralpom.core_factory import CORE_REGISTRY, resolve_core_class
from neuralpom.cores.implicit_p0_fixed.time_loop import barotropic_endpoint_transport
from neuralpom.cores.implicit_p0_fixed.autograd import strain_magnitude
from neuralpom.hybrid import build_initialized_core
from neuralpom.hybrid.sparse_physics import SparsePhysics
from neuralpom.v1.checkpoint import capture, restore
from neuralpom.v1.core_spec import core_spec


def make_core():
    core = build_initialized_core(resolve_core_class("implicit_p0_fixed"), torch.device("cpu"), 10, 10)
    core.time = core.time0 = 1000.0
    return core


def make_explicit_core():
    spec = core_spec("explicit")
    core = build_initialized_core(spec.core_class, torch.device("cpu"), 10, 10)
    spec.configure(core)
    core.time = core.time0 = 1000.0
    return core


class FixedCoreRuntimeTests(unittest.TestCase):
    def test_strict_vertical_rejects_misleading_kernel_selection(self):
        for mode in ("strict_vertical", "strict_vertical_compiled"):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "only proft"):
                    SparsePhysics("cuda", acceleration=mode, compiled_kernels=("profq",))
                with self.assertRaisesRegex(ValueError, "only proft"):
                    SparsePhysics("cuda", acceleration=mode, compiled_kernels=("profq", "proft"))
                physics = SparsePhysics("cuda", acceleration=mode, compiled_kernels=("proft",))
                self.assertEqual(physics.compiled_kernels, ("proft",))

    def test_registry_and_fixed_transport(self):
        self.assertEqual(set(CORE_REGISTRY), {"implicit_p0_fixed", "explicit"})
        self.assertIsNotNone(resolve_core_class("explicit"))
        for name in ("implicit",):
            with self.assertRaises(ValueError):
                resolve_core_class(name)
        velocity = torch.tensor([0.2, -0.3])
        depth = torch.tensor([10.0, 20.0])
        transport = barotropic_endpoint_transport(velocity, 2 * depth, velocity, 2 * depth)
        torch.testing.assert_close(transport, velocity * depth)

    def test_full_bptt_checkpoint_matches_direct_gradient(self):
        baseline = capture(make_core())
        results = []
        for checkpoint_physics in (False, True):
            core = make_core()
            restore(core, baseline)
            parameter = torch.tensor(0.01, requires_grad=True)
            pattern = torch.linspace(-1, 1, core.t.numel()).reshape_as(core.t)
            core.t = core.t + parameter * pattern
            core.tb = core.tb + parameter * pattern
            physics = SparsePhysics("cpu", chunk_steps=1, checkpoint_physics=checkpoint_physics)
            physics.advance(core, steps=2, differentiable=True)
            loss = core.t.square().mean() + core.u.square().mean()
            gradient, = torch.autograd.grad(loss, parameter)
            results.append((core.t.detach(), gradient))
        self.assertGreater(abs(float(results[0][1])), 1e-6)
        for left, right in zip(*results):
            torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-8)

    def test_explicit_forward_gradient_and_complete_restart(self):
        baseline = capture(make_explicit_core())
        outputs = []
        for checkpoint_physics in (False, True):
            core = make_explicit_core()
            restore(core, baseline)
            amplitude = torch.tensor(0.01, requires_grad=True)
            pattern = torch.linspace(-1, 1, core.t.numel()).reshape_as(core.t)
            core.t = core.t + amplitude * pattern
            core.tb = core.tb + amplitude * pattern
            physics = SparsePhysics("cpu", core_name="explicit", compiled_kernels=(),
                                    chunk_steps=2, checkpoint_physics=checkpoint_physics)
            physics.advance(core, steps=2, differentiable=True)
            gradient, = torch.autograd.grad(core.t.square().mean() + core.u.square().mean(), amplitude)
            outputs.append((core.t.detach(), gradient))
        self.assertTrue(bool(torch.isfinite(outputs[0][1])))
        self.assertGreater(abs(float(outputs[0][1])), 1e-6)
        for left, right in zip(*outputs):
            torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-8)

        physics = SparsePhysics("cpu", core_name="explicit", compiled_kernels=())
        direct = make_explicit_core()
        physics.advance(direct, steps=2)
        restarted = make_explicit_core()
        restore(restarted, capture(direct))
        physics.advance(direct, steps=2)
        physics.advance(restarted, steps=2)
        state, replay = capture(direct), capture(restarted)
        self.assertEqual(state["tensors"].keys(), replay["tensors"].keys())
        for name in state["tensors"]:
            torch.testing.assert_close(state["tensors"][name], replay["tensors"][name],
                                       rtol=0, atol=0, msg=name)
        self.assertEqual(direct.time, restarted.time)

    def test_strict_vertical_composes_with_full_state_checkpoint(self):
        baseline = capture(make_core())
        results = []
        for mode in ("eager", "strict_vertical"):
            core = make_core()
            restore(core, baseline)
            amplitude = torch.tensor(0.01, requires_grad=True)
            pattern = torch.linspace(-1, 1, core.t.numel()).reshape_as(core.t)
            core.t = core.t + amplitude * pattern
            core.tb = core.tb + amplitude * pattern
            SparsePhysics("cpu", acceleration=mode, chunk_steps=1).advance(core, steps=2, differentiable=True)
            gradient, = torch.autograd.grad(core.t.square().mean() + core.u.square().mean(), amplitude)
            results.append((capture(core), gradient))
        for name, value in results[0][0]["tensors"].items():
            torch.testing.assert_close(value, results[1][0]["tensors"][name], rtol=0, atol=0, msg=name)
        torch.testing.assert_close(results[0][1], results[1][1], rtol=1e-6, atol=1e-8)

    def test_serialized_restart_reproduces_every_state_tensor(self):
        core = make_core()
        physics = SparsePhysics("cpu", chunk_steps=1)
        physics.advance(core, steps=2)
        buffer = io.BytesIO()
        torch.save(capture(core), buffer)
        buffer.seek(0)
        restarted = make_core()
        restore(restarted, torch.load(buffer, weights_only=False))
        physics.advance(core, steps=2)
        physics.advance(restarted, steps=2)
        direct, replay = capture(core), capture(restarted)
        self.assertEqual(direct["tensors"].keys(), replay["tensors"].keys())
        for name in direct["tensors"]:
            torch.testing.assert_close(direct["tensors"][name], replay["tensors"][name],
                                       rtol=0, atol=0, msg=name)
        self.assertEqual(core.time, restarted.time)

    def test_zero_strain_gradient_preserves_exact_forward(self):
        value = torch.tensor([0.0, 1e-20, 0.1, 2.0], dtype=torch.float64, requires_grad=True)
        actual = strain_magnitude(value.square())
        torch.testing.assert_close(actual, torch.sqrt(value.square()), rtol=0, atol=0)
        actual.sum().backward()
        torch.testing.assert_close(value.grad, torch.tensor([0., 1., 1., 1.], dtype=torch.float64))
        self.assertTrue(torch.autograd.gradcheck(strain_magnitude, (torch.tensor([0.1, 1.], dtype=torch.float64, requires_grad=True),)))
        core = make_core()
        amplitude = torch.tensor(0., requires_grad=True)
        pattern = torch.sin(torch.arange(core.u.numel()).reshape_as(core.u) * 0.1)
        core.u = core.u + amplitude * pattern
        core.ub = core.ub + amplitude * pattern
        SparsePhysics("cpu", chunk_steps=1).advance(core, steps=1, differentiable=True)
        (core.u.square().mean() + core.t.square().mean()).backward()
        self.assertTrue(bool(torch.isfinite(amplitude.grad)))

    def test_eta_adjoint_matches_transpose_and_directional_difference(self):
        core = make_core()
        for name, value in list(vars(core).items()):
            if torch.is_tensor(value) and value.is_floating_point():
                setattr(core, name, value.double())
        core._accel_cache_ready = False
        core._spectral_eta_preconditioner_cache = None
        core.eta_solver_rel_tol = 1e-11
        depth = core.d.detach().clone().requires_grad_(True)
        rhs = torch.sin(torch.arange(100, dtype=torch.float64)).reshape(10, 10).requires_grad_(True)
        weight = torch.cos(torch.arange(100, dtype=torch.float64)).reshape(10, 10)
        weight = weight.clone()
        weight[[0, -1], :] = 0
        weight[:, [0, -1]] = 0
        solution = core.solve_implicit_eta_cg(rhs, depth, max_iter=200, tol=1e-14,
                                            print_cg_residual=False, initial_guess=torch.zeros_like(rhs))
        rhs_grad, depth_grad = torch.autograd.grad((solution * weight).sum(), (rhs, depth))
        columns = []
        for index in range(64):
            basis = torch.zeros_like(rhs)
            basis[1 + index // 8, 1 + index % 8] = 1
            columns.append(core.apply_elliptic_operator(basis, depth.detach())[1:-1, 1:-1].reshape(-1))
        matrix = torch.stack(columns, dim=1)
        expected = torch.linalg.solve(matrix.T, weight[1:-1, 1:-1].reshape(-1))
        torch.testing.assert_close(rhs_grad[1:-1, 1:-1].reshape(-1), expected, rtol=1e-8, atol=1e-10)
        residual = matrix @ solution.detach()[1:-1, 1:-1].reshape(-1) - rhs.detach()[1:-1, 1:-1].reshape(-1)
        self.assertLess(float(residual.norm() / rhs.detach().norm()), 1e-9)
        direction = torch.sin(torch.arange(100, dtype=torch.float64) * 0.3).reshape(10, 10)
        epsilon = 0.01
        losses = []
        with torch.no_grad():
            for sign in (-1, 1):
                eta = core.solve_implicit_eta_cg(rhs.detach(), depth.detach() + sign * epsilon * direction,
                    max_iter=200, tol=1e-14, print_cg_residual=False, initial_guess=torch.zeros_like(rhs))
                losses.append((eta * weight).sum())
        finite_difference = (losses[1] - losses[0]) / (2 * epsilon)
        torch.testing.assert_close((depth_grad * direction).sum(), finite_difference, rtol=1e-5, atol=1e-10)

    def test_large_rhs_adjoint_stays_finite_and_linear(self):
        core = make_core()
        gradient = torch.sin(torch.arange(100, dtype=torch.float32)).reshape(10, 10)
        small = core._solve_implicit_eta_adjoint(gradient, core.d, max_iter=200, tol=1e-12)
        large = core._solve_implicit_eta_adjoint(gradient * 1e20, core.d, max_iter=200, tol=1e-12)
        self.assertTrue(bool(torch.isfinite(large).all()))
        torch.testing.assert_close(large / 1e20, small, rtol=1e-4, atol=1e-6)

    def test_partial_final_checkpoint_chunk_advances_exact_time(self):
        core = make_core()
        SparsePhysics("cpu", chunk_steps=2).advance(core, steps=3, differentiable=True)
        self.assertAlmostEqual(core.time, 1000 + 1800 / 86400, places=11)


if __name__ == "__main__":
    unittest.main()

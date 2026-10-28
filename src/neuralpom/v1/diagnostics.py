"""Daily field errors, tracer budgets, density profiles, and native-grid spectra."""
from __future__ import annotations

import numpy as np
import torch

from neuralpom.hybrid.sparse_coupling import LEVELS, eos

FIELDS = ("u", "v", "t", "s", "el")


def numpy_fields(state):
    return {key: value.detach().cpu().numpy().copy() if torch.is_tensor(value)
            else np.asarray(value).copy() for key, value in state.items()}


def velocity_spectra(prediction, reference, dz, wet_u, wet_v):
    """Additive spectral energies, before any case averaging or ratios.

    Frequencies are cycles per grid interval, on native staggered velocity
    grids. A common Hann window and wet mask are used for both states. The
    last representation is the sigma-weighted depth-mean velocity.
    """
    dz = np.asarray(dz, dtype=np.float64)[:LEVELS]
    dz = dz / dz.sum()
    nx, ny = np.asarray(prediction["u"]).shape[:2]
    if min(nx, ny) < 4:
        raise ValueError("Spectra need at least four cells per horizontal axis")
    window = np.hanning(nx)[:, None] * np.hanning(ny)[None, :]

    def transform(state):
        transformed = []
        for field, wet in (("u", wet_u), ("v", wet_v)):
            values = np.asarray(state[field], dtype=np.float64)[..., :LEVELS].transpose(2, 0, 1)
            values = np.concatenate((values, (values * dz[:, None, None]).sum(0, keepdims=True)))
            weights = window * np.asarray(wet)
            if weights.sum() <= 0:
                raise ValueError("Spectrum has no wet cells")
            mean = (values * weights).sum((-2, -1), keepdims=True) / weights.sum()
            values = (values - mean) * weights / np.sqrt(np.mean(weights**2))
            transformed.append(np.fft.fft2(values, axes=(-2, -1), norm="forward"))
        return np.stack(transformed)

    p, r = transform(prediction), transform(reference)
    radius = np.hypot(np.fft.fftfreq(nx)[:, None], np.fft.fftfreq(ny)[None, :])
    edges = np.linspace(0, np.sqrt(.5), min(nx, ny) // 2 + 1)
    bins = np.searchsorted(edges, radius, side="right") - 1
    bins = bins.clip(0, len(edges) - 2)
    result = {"wavenumber": (.5 * (edges[:-1] + edges[1:])).tolist(),
              "units": "cycles per grid interval", "sigma_weights": dz.tolist(),
              "representations": [f"layer_{i}" for i in range(LEVELS)] + ["depthmean"]}
    for name, energy in (
        ("prediction", .5 * np.sum(np.abs(p)**2, axis=0)),
        ("reference", .5 * np.sum(np.abs(r)**2, axis=0)),
        ("error", .5 * np.sum(np.abs(p - r)**2, axis=0)),
        ("cross", .5 * np.real(np.sum(p * np.conj(r), axis=0))),
    ):
        result[name] = np.stack([energy[:, bins == i].sum(-1) for i in range(len(edges) - 1)], -1).tolist()
    return result


@torch.no_grad()
def state_diagnostics(core, prediction, reference):
    device = core.t.device
    p = {key: torch.as_tensor(value, device=device, dtype=torch.float64) for key, value in prediction.items()}
    r = {key: torch.as_tensor(reference[key], device=device, dtype=torch.float64) for key in FIELDS}
    area = core.art.double() * core.fsm.double()
    reference_surface = torch.as_tensor(reference.get("et", reference["el"]), device=device, dtype=torch.float64)
    depth = core.h.double() + reference_surface
    if not bool((depth[core.fsm > .5] > 0).all()):
        raise ValueError("Nonpositive reference wet depth")
    sigma = core.dz[:LEVELS].double()
    volume = (area * depth)[..., None] * sigma
    layer_volume = volume.sum((0, 1))
    metrics, profiles = {}, {"sigma": core.zz[:LEVELS].cpu().tolist()}
    for field in FIELDS:
        weights = area if field == "el" else volume
        if field in ("u", "v"):
            mask = core.dum if field == "u" else core.dvm
            weights = (core.art.double() * mask.double() * depth)[..., None] * sigma
        predicted = p[field] if field == "el" else p[field][..., :LEVELS]
        target = r[field] if field == "el" else r[field][..., :LEVELS]
        if not bool(torch.isfinite(predicted).all()):
            raise FloatingPointError(f"Nonfinite prediction {field}")
        error = predicted - target
        mse = (error.square() * weights).sum() / weights.sum()
        metrics[f"mse/{field}"] = float(mse)
        metrics[f"rmse/{field}"] = float(mse.sqrt())
        metrics[f"bias/{field}"] = float((error * weights).sum() / weights.sum())
        if field in ("t", "s"):
            mean_error = (error * weights).sum((0, 1)) / layer_volume
            global_error = (error * weights).sum() / volume.sum()
            parts = {"global_mean": global_error.square(),
                     "profile_shape": ((mean_error - global_error).square() * layer_volume).sum() / volume.sum(),
                     "horizontal": ((error - mean_error).square() * weights).sum() / volume.sum()}
            if not torch.allclose(sum(parts.values()), mse, atol=1e-12, rtol=1e-6):
                raise AssertionError("Tracer error decomposition does not close")
            metrics.update({f"decomposition/{field}/{key}": float(value) for key, value in parts.items()})
            metrics[f"inventory/{field}/prediction_common_volume"] = float((predicted * volume).sum())
            physical_volume = (area * (core.h.double() + core.et.double()))[..., None] * sigma
            metrics[f"inventory/{field}/prediction_physical_volume"] = float((predicted * physical_volume).sum())
            metrics[f"inventory/{field}/reference"] = float((target * volume).sum())
            profiles[f"{field}/prediction"] = ((predicted * volume).sum((0, 1)) / layer_volume).cpu().tolist()
            profiles[f"{field}/reference"] = ((target * volume).sum((0, 1)) / layer_volume).cpu().tolist()
            profiles[f"{field}/mse"] = ((error.square() * volume).sum((0, 1)) / layer_volume).cpu().tolist()
    # The actual core EOS supplies in-situ density; no proxy polynomial is used.
    rho = 1000 + core.rhoref * eos(core, p["t"].float(), p["s"].float())[..., :LEVELS].double()
    truth_rho = 1000 + core.rhoref * eos(core, r["t"].float(), r["s"].float())[..., :LEVELS].double()
    metrics["density/mse"] = float(((rho - truth_rho).square() * volume).sum() / volume.sum())
    profiles["rho/prediction"] = ((rho * volume).sum((0, 1)) / layer_volume).cpu().tolist()
    profiles["rho/reference"] = ((truth_rho * volume).sum((0, 1)) / layer_volume).cpu().tolist()
    spacing = -depth[..., None] * (core.zz[1:LEVELS] - core.zz[:LEVELS-1]).double()
    slope = (rho[..., 1:] - rho[..., :-1]) / spacing.clamp_min(1e-12)
    weights = .5 * (volume[..., 1:] + volume[..., :-1])
    metrics["density/negative_vertical_slope_fraction"] = float(((slope < 0) * weights).sum() / weights.sum())
    metrics["volume/reference"] = float(volume.sum())
    metrics["volume/prediction"] = float((area * (core.h.double() + core.et.double())).sum())
    spectral = velocity_spectra(numpy_fields(p), numpy_fields(r), sigma.cpu().numpy(),
                                core.dum.cpu().numpy(), core.dvm.cpu().numpy())
    return metrics, profiles, spectral

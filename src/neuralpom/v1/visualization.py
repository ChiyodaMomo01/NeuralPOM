"""Render evaluation artifacts without selecting or modifying model weights."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .diagnostics import FIELDS


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _prefix_suffix(metadata):
    if metadata.get('selection') != 'validation_prefix':
        return ''
    windows = metadata.get('validated_windows')
    return ' | Validation prefix' + (f' ({windows} windows)' if windows is not None else '')


def render_comparison(path):
    """Plot paired skill without averaging ratios across origins."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = Path(path)
    comparison = json.loads(path.read_text())
    cases = {}
    for row in comparison["daily"]:
        cases.setdefault((row["member"], row["start_day"]), []).append(row)
    figure, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for case, rows in cases.items():
        rows = sorted(rows, key=lambda row: row["lead_day"])
        for axis, field in zip(axes.flat, FIELDS):
            values = [row["rmse_ratio"][f"post/{field}"] for row in rows]
            axis.plot([row["lead_day"] for row in rows],
                      [np.nan if value is None else value for value in values],
                      linewidth=1, marker="." if len(rows) == 1 else None,
                      alpha=.7, label=f"m{case[0]}, day {case[1]}")
            axis.set(xlabel="Lead day", ylabel=f"{field}: model / physics RMSE")
            axis.grid(alpha=.2)
    for axis in list(axes.flat)[:len(FIELDS)]:
        axis.axhline(1., color="black", linestyle="--", linewidth=.8)
        low, high = axis.get_ylim()
        axis.set_ylim(min(low, .99), max(high, 1.01))
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        if comparison.get("days") == 1:
            axis.set(xlim=(.5, 1.5), xticks=[1])
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="upper left", frameon=False, fontsize=8)
    figure.suptitle("Paired RMSE ratio | below 1 means lower error" + _prefix_suffix(comparison))
    output = path.parent / "figures" / "physics_rmse_ratio.png"
    output.parent.mkdir(exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def render_evaluation(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullFormatter
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text())
    prefix_suffix = _prefix_suffix(summary)
    geometry = {}
    if (root / "geometry.npz").exists():
        with np.load(root / "geometry.npz") as archive:
            geometry = {name: archive[name].copy() for name in ("fsm", "dum", "dvm") if name in archive}
    cases = [case for case in summary["cases"] if case["completed_days"] > 0]
    if not cases:
        return []
    output = root / "figures"
    output.mkdir(exist_ok=True)
    images = []
    figure, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for case in cases:
        rows = _rows(case["metrics_file"])
        lead = [row["lead_day"] for row in rows]
        for index, field in enumerate(FIELDS):
            axis = axes.flat[index]
            axis.plot(lead, [row["metrics"][f"post/rmse/{field}"] for row in rows],
                      "-", color="#2563a6", alpha=.7, linewidth=1,
                      marker="." if len(lead) == 1 else None)
            axis.set(xlabel="Lead day", ylabel=f"{field} RMSE")
            axis.grid(alpha=.2)
            if len(lead) == 1:
                axis.set(xlim=(lead[0] - .5, lead[0] + .5), xticks=lead)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0, .9, f"Complete cases: {summary['completed_cases']}/{summary['requested_cases']}\n"
                     "Blue: after correction versus reference\nEach line is one initial state",
                     va="top", transform=axes.flat[-1].transAxes)
    figure.suptitle(root.name + prefix_suffix)
    path = output / "daily_rmse.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    images.append(path)
    intraday_cases = [(case, _rows(Path(case["metrics_file"]).parent / "intraday.jsonl"))
                      for case in cases if (Path(case["metrics_file"]).parent / "intraday.jsonl").exists()]
    if intraday_cases:
        figure, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
        for case, rows in intraday_cases:
            for hour in (1, 2, 3, 4):
                hourly = [row for row in rows if row["within_day_hour"] == hour]
                for axis, field in zip(axes.flat, FIELDS):
                    axis.plot([row["lead_seconds"] / 86400 for row in hourly],
                              [row["metrics"][f"post/rmse/{field}"] for row in hourly],
                              marker=".", markersize=3, linewidth=.8, alpha=.65,
                              color=plt.get_cmap("tab10")(hour - 1))
                    axis.set(xlabel="Lead day", ylabel=f"{field} RMSE after correction")
                    axis.grid(alpha=.2)
        from matplotlib.lines import Line2D
        axes.flat[-1].axis("off")
        axes.flat[-1].legend(handles=[Line2D([0], [0], color=plt.get_cmap("tab10")(hour - 1),
                                                 label=f"Hour {hour} of each day") for hour in (1, 2, 3, 4)],
                             loc="upper left", frameon=False)
        figure.suptitle(root.name + " | first four hours of each day" + prefix_suffix)
        path = output / "intraday_rmse.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        images.append(path)
    # Each origin remains separately identifiable; incomplete trajectories show
    # their actual last available day, never a padded nominal terminal date.
    for case in cases:
        directory = Path(case["metrics_file"]).parent
        profile = _rows(directory / "profiles.jsonl")[-1]
        spectrum = _rows(directory / "spectra.jsonl")[-1]
        label = f"m{case['member']}_d{case['start_day']}_lead{case['completed_days']}"
        figure, axes = plt.subplots(1, 3, figsize=(11, 4), constrained_layout=True)
        for axis, field in zip(axes, ("t", "s", "rho")):
            sigma = profile["post"]["sigma"]
            axis.plot(profile["post"][f"{field}/reference"], sigma, color="black", label="Reference")
            axis.plot(profile["post"][f"{field}/prediction"], sigma, color="#2563a6", label="After correction")
            axis.set(xlabel=field, ylabel="Sigma coordinate")
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
            axis.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        figure.suptitle(label + prefix_suffix)
        path = output / f"{label}_profiles.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        images.append(path)
        figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
        data = spectrum["post"]
        sigma = np.asarray(data["sigma_weights"])
        for column, representation in enumerate(("surface", "sigma_weighted", "depthmean")):
            axis = axes[0, column]
            for name, color, style in (("reference", "black", "-"), ("prediction", "#2563a6", "-"),
                                       ("error", "#b54708", "--")):
                values = np.asarray(data[name])
                energy = values[0] if representation == "surface" else values[-1] if representation == "depthmean" else (values[:-1] * sigma[:, None]).sum(0)
                axis.loglog(data["wavenumber"], np.maximum(energy, 1e-30), style, color=color, label=name)
            axis.set(title=representation, xlabel="Cycles per grid interval", ylabel="Windowed bin energy (m²/s²)")
            axis.grid(alpha=.2)
            values = np.asarray(data["cross"])
            cross = values[0] if representation == "surface" else values[-1] if representation == "depthmean" else (values[:-1] * sigma[:, None]).sum(0)
            axes[1, column].semilogx(data["wavenumber"], cross, color="#577d46")
            axes[1, column].axhline(0, color="0.6", linewidth=.8)
            axes[1, column].set(xlabel="Cycles per grid interval", ylabel="Signed cross energy (m²/s²)")
            axes[1, column].grid(alpha=.2)
        axes[0, 0].legend(fontsize=8)
        ticks = np.geomspace(data["wavenumber"][0], data["wavenumber"][-1], 4)
        for axis in axes.flat:
            axis.xaxis.set_major_locator(FixedLocator(ticks))
            axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2g}"))
            axis.xaxis.set_minor_formatter(NullFormatter())
        figure.suptitle(label + prefix_suffix)
        path = output / f"{label}_spectra.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        images.append(path)
        day = case["completed_days"]
        states = {}
        for name in ("reference", "post"):
            with np.load(directory / f"day{day:03d}_{name}.npz") as archive:
                states[name] = {field: archive[field].copy() for field in FIELDS}
        figure, axes = plt.subplots(3, 5, figsize=(15, 8), constrained_layout=True)
        for col, field in enumerate(FIELDS):
            values = {name: data[field] if field == "el" else data[field][..., 0] for name, data in states.items()}
            mask_name = "dum" if field == "u" else "dvm" if field == "v" else "fsm"
            wet = geometry.get(mask_name, np.ones_like(values["reference"])) > 0
            values = {name: np.ma.masked_where(~wet, value) for name, value in values.items()}
            low = min(float(value.min()) for value in values.values())
            high = max(float(value.max()) for value in values.values())
            for row, name in enumerate(("reference", "post")):
                im = axes[row, col].imshow(values[name].T, origin="lower", vmin=low, vmax=high, cmap="viridis")
                axes[row, col].set(title=f"{name}: {field}", xticks=[], yticks=[])
            figure.colorbar(im, ax=axes[:2, col].tolist(), shrink=.6)
            error = values["post"] - values["reference"]
            limit = max(float(np.abs(error).max()), 1e-12)
            im = axes[2, col].imshow(error.T, origin="lower", vmin=-limit, vmax=limit, cmap="RdBu_r")
            axes[2, col].set(title="Post minus reference", xticks=[], yticks=[])
            figure.colorbar(im, ax=axes[2, col], shrink=.8)
        figure.suptitle(label + " | surface fields" + prefix_suffix)
        path = output / f"{label}_fields.png"
        figure.savefig(path, dpi=140)
        plt.close(figure)
        images.append(path)
    return images

"""Standalone AI exports using the common physical-unit diagnostic keys."""
from pathlib import Path
import csv
import numpy as np
from neuralpom.evaluation.protocol import read_json, atomic_json, FIELDS, case_root
from neuralpom.evaluation.analysis import records


def summarize(root):
    root = Path(root)
    manifest = read_json(root/"manifest.json")
    variant, cases, days = manifest["variant"], manifest["cases"], manifest["days"]
    rows, statuses = [], []
    for case in cases:
        directory = case_root(root, variant["id"], case)
        status = read_json(directory/"status.json")
        daily = records(directory/"daily.jsonl", status["completed_days"])
        if [r["lead_day"] for r in daily] != list(range(1, status["completed_days"]+1)):
            raise ValueError("Missing committed AI daily scores")
        statuses.append(dict(case=case, **status))
        rows.append((case, daily))
    complete = [(case, daily) for case, daily in rows if len(daily) == days]
    result = dict(variant=variant, days=days, requested_cases=len(cases), paired_full_windows=len(complete),
        statuses=statuses, physical_forward_steps=0, reference_feedback=False,
        primary_metric="same-core spatially weighted physical-unit MSE",
        mean_mse={field: float(np.mean([row["metrics"][f"mse/{field}"]
            for _, daily in complete for row in daily])) if complete else None for field in FIELDS},
        conditional_on_full_window=True)
    atomic_json(root/"summary.json", result)
    with (root/"daily_mse.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant", "member", "start_day", "lead_day", *[f"mse_{f}" for f in FIELDS]])
        for case, daily in rows:
            for row in daily:
                writer.writerow([variant["id"], case["member"], case["start_day"], row["lead_day"],
                    *[row["metrics"][f"mse/{f}"] for f in FIELDS]])
    keys = sorted({key for _, daily in rows for row in daily for key in row["metrics"]})
    with (root/"daily_metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant", "member", "start_day", "lead_day", *keys])
        for case, daily in rows:
            for row in daily:
                writer.writerow([variant["id"], case["member"], case["start_day"], row["lead_day"],
                    *[row["metrics"].get(key) for key in keys]])
    return result

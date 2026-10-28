"""Frozen, paired evaluation with daily fields and reusable diagnostics."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from neuralpom.models.corrector_net import CorrectorNet
from .checkpoint import atomic_json, distributed
from .diagnostics import FIELDS, numpy_fields, state_diagnostics
from .data import NOMINAL_CLOCK_TOL_DAYS
from .runtime import frozen_rollout


def _append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def _save_fields(path, fields, time):
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **numpy_fields(fields), time=np.float64(time))
    temporary.replace(path)


def summarize_cases(cases, days):
    """Keep partial runs visible; aggregate skill only over complete cases."""
    complete = [case for case in cases if case["completed_days"] == days and case["status"] == "complete"]
    result = {"requested_cases": len(cases), "completed_cases": len(complete),
              "completion_fraction": len(complete) / len(cases) if cases else 0.,
              "days": days, "cases": cases, "periods": {}, "evaluation_field": "post_correction"}
    for name, lower, upper in (("first50", 1, min(50, days)), ("whole", 1, days),
                               ("last50", max(1, days - 49), days)):
        records = []
        for case in complete:
            records.extend(json.loads(line) for line in Path(case["metrics_file"]).read_text().splitlines()
                           if line.strip())
        records = [row for row in records if lower <= row["lead_day"] <= upper]
        result["periods"][name] = {}
        for field in FIELDS:
            values = [row["metrics"][f"post/mse/{field}"] for row in records]
            if values:
                result["periods"][name][f"post/rmse/{field}"] = float(np.sqrt(np.mean(values)))
    return result


def _phase(function, device):
    """A local error becomes a shared error before the next DDP operation."""
    if not distributed():
        return function()
    result, failure = None, None
    try:
        result = function()
    except Exception as error:
        failure = f'{type(error).__name__}: {error}'
    flag = torch.tensor(int(failure is not None), device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if int(flag):
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, failure)
        raise RuntimeError('Distributed evaluation failed: ' +
                           '; '.join(f'rank {rank}: {error}' for rank, error in enumerate(errors) if error))
    return result


def _check_time(value, expected):
    value = float(value)
    if not math.isfinite(value) or abs(value - expected) > NOMINAL_CLOCK_TOL_DAYS:
        raise ValueError(f'Evaluation clock {value} differs from expected {expected}')
    return value


def evaluate(config, runtime, weights, days, logger, baseline=False):
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed() else (0, 1)
    def phase(function):
        return _phase(function, runtime.device)

    def preflight():
        if isinstance(days, bool) or int(days) != days or days <= 0:
            raise ValueError('Evaluation duration must be a positive integer')
        weight_metadata = dict(selection=None, step=None, validated_windows=None, supervision=None, posttrain_gradient_mode=None)
        if baseline:
            model, stage = torch.nn.Identity().to(runtime.device), 'physics'
        else:
            if weights is None:
                raise ValueError('Frozen evaluation requires stage weights')
            checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
            weight_metadata = {key: checkpoint.get(key) for key in weight_metadata}
            if checkpoint['contract'] != runtime.contract():
                raise ValueError('Frozen weights and evaluation state contract differ')
            stage = checkpoint['stage']
            # Historical two/three-step weights remain readable for comparison.
            if stage not in ('pretrain', 'midtrain2', 'midtrain3', 'midtrain4', 'posttrain'):
                raise ValueError('Unknown frozen training stage')
            if days == 300 and stage != 'posttrain':
                raise ValueError('The registered 300-day model evaluation uses Post-train weights')
            model = CorrectorNet(101, 60, config['model']['hidden_dim']).to(runtime.device)
            model.load_state_dict(checkpoint['model'], strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        frozen = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        settings = config['split']
        cases = [(member, start) for member in settings['evaluation_members']
                 for start in settings['evaluation_starts']]
        if not cases or len(cases) != len(set(cases)):
            raise ValueError('Evaluation needs a nonempty, unique case list')
        for member, start in cases:
            if (not isinstance(member, int) or not 0 <= member < 8 or
                    not isinstance(start, int) or not settings['evaluation_start_day'] <= start or
                    start + days > settings['evaluation_end_day']):
                raise ValueError('Evaluation labels exceed the registered evaluation split')
        return model, stage, frozen, cases, weight_metadata
    model, stage, frozen, cases, weight_metadata = phase(preflight)
    if distributed():
        descriptor = {'stage': stage, 'cases': cases, 'days': days, 'contract': runtime.contract(),
                      'frequency': config['frequency'], 'run_root': str(config['run_root']),
                      'weights': str(weights) if weights else None, **weight_metadata}
        descriptors = [None] * world
        dist.all_gather_object(descriptors, descriptor)
        if any(value != descriptors[0] for value in descriptors):
            raise ValueError('Evaluation ranks have different cases, weights, or state contracts')
    days = int(days)
    root = Path(config['run_root']) / 'evaluation' / f'{stage}_{days}d'

    def prepare_output():
        if rank != 0:
            return
        root.mkdir(parents=True, exist_ok=True)
        if (root / 'manifest.json').exists():
            raise FileExistsError(f'Evaluation already exists: {root}')
        geometry = {key: getattr(runtime.core, key) for key in
                    ('fsm', 'dum', 'dvm', 'h', 'art', 'aru', 'arv', 'dx', 'dy', 'dz', 'zz')
                    if hasattr(runtime.core, key)}
        geometry_path = root / 'geometry.npz'
        temporary = root / 'geometry.tmp.npz'
        np.savez_compressed(temporary, **numpy_fields(geometry))
        temporary.replace(geometry_path)
        atomic_json(root / 'manifest.json', {'contract': runtime.contract(), 'stage': stage,
                    'weights': str(weights) if weights else None, 'days': days, 'cases': cases,
                    'frequency': config['frequency'], 'frozen': True, 'geometry': 'geometry.npz',
                    'spectrum_units': 'cycles per grid interval',
                    'evaluation_field': 'post_correction', **weight_metadata})
    phase(prepare_output)
    local = []
    slots = math.ceil(len(cases) / world)
    for slot in range(slots):
        case_index = slot * world + rank
        active = case_index < len(cases)
        context = {}
        def start_case():
            if not active:
                return
            member, start = cases[case_index]
            output = root / f'm{member}_day{start:04d}'
            output.mkdir(parents=True, exist_ok=True)
            metrics_file = output / 'daily.jsonl'
            if metrics_file.exists() and metrics_file.stat().st_size:
                raise FileExistsError(f'Evaluation case already contains results: {output}')
            context.update(member=member, start=start, output=output, intraday_count=0,
                           status={'member': member, 'start_day': start, 'completed_days': 0,
                                   'status': 'running', 'metrics_file': str(metrics_file)})
            def record_intraday(row):
                if config['frequency'] != '1h':
                    raise ValueError('Intraday scoring is reserved for the 1 h setting')
                count = context['intraday_count']
                expected_lead = (count // 4) * 86400 + (count % 4 + 1) * 3600
                lead = row['lead_seconds']
                if lead != expected_lead or lead > days * 86400:
                    raise ValueError('Intraday diagnostics require hours 1, 2, 3, 4 of each day in order')
                hour = (lead % 86400) // 3600
                if row['within_day_hour'] != hour:
                    raise ValueError('Intraday hour and lead do not match')
                time = _check_time(row['time'], start + lead / 86400.)
                metrics = {key: value for key, value in row['metrics'].items() if key.startswith('post/')}
                # Hourly reference files contain el but no et; diagnostics use
                # that observed surface for common-volume error weighting.
                values, _, _ = state_diagnostics(runtime.core, row['post'], row['reference'])
                metrics.update({f'post/{key}': value for key, value in values.items()})
                if not all(math.isfinite(float(value)) for value in metrics.values()):
                    raise FloatingPointError('Nonfinite intraday evaluation metric')
                record = {'lead_seconds': lead, 'time': time, 'within_day_hour': hour,
                          'metrics': metrics, 'corrections': row.get('correction_diagnostics', {})}
                if 'reference_time' in row:
                    record['reference_time'] = _check_time(row['reference_time'], start + lead / 86400.)
                _append(output / 'intraday.jsonl', record)
                context['intraday_count'] += 1
            callback = record_intraday if config['frequency'] == '1h' else None
            context['generator'] = iter(frozen_rollout(runtime, model, member, start, days,
                                                       baseline=baseline, on_intraday=callback))
        phase(start_case)
        def one_day(expected_day):
            if not active or context['status']['status'] != 'running':
                return
            member, start, output = context['member'], context['start'], context['output']
            status = context['status']
            try:
                try:
                    row = next(context['generator'])
                except StopIteration as error:
                    raise ValueError('Frozen rollout ended before the requested horizon') from error
                if row['lead_day'] != expected_day:
                    raise ValueError('Frozen rollout returned an incomplete or repeated lead sequence')
                time = _check_time(row['time'], start + expected_day)
                reference = runtime.data.load_initial(member, start + expected_day, map_location=runtime.device)
                reference_time = _check_time(reference['time'], start + expected_day)
                metrics = {key: value for key, value in row['metrics'].items() if key.startswith('post/')}
                values, profile, spectrum = state_diagnostics(runtime.core, row['post'], reference)
                metrics.update({f'post/{key}': value for key, value in values.items()})
                if not all(math.isfinite(float(value)) for value in metrics.values()):
                    raise FloatingPointError('Nonfinite evaluation metric')
                _save_fields(output / f'day{expected_day:03d}_post.npz', row['post'], time)
                _save_fields(output / f'day{expected_day:03d}_reference.npz',
                             {key: reference[key] for key in FIELDS}, reference_time)
                record = {'lead_day': expected_day, 'time': time, 'reference_time': reference_time,
                          'metrics': metrics, 'corrections': row.get('correction_diagnostics', {})}
                _append(output / 'profiles.jsonl', {'lead_day': expected_day, 'post': profile})
                _append(output / 'spectra.jsonl', {'lead_day': expected_day, 'post': spectrum})
                _append(status['metrics_file'], record)
                status['completed_days'] = expected_day
                atomic_json(output / 'status.json', status)
                if rank == 0:
                    logger.log({f'evaluation/m{member}_d{start}/{key}': value for key, value in metrics.items()},
                               slot * days + expected_day)
            except FloatingPointError as error:
                actual = float(runtime.core.time)
                actual = actual if math.isfinite(actual) else None
                status.update(status='nonfinite', first_anomaly_day=status['completed_days'] + 1,
                              first_anomaly_time=actual,
                              first_anomaly_lead_seconds=(actual - start) * 86400. if actual is not None else None,
                              error=str(error))
        try:
            for day in range(1, days + 1):
                phase(lambda: one_day(day))
            def finish_case():
                if not active:
                    return
                status = context['status']
                if status['status'] == 'running':
                    try:
                        next(context['generator'])
                    except StopIteration:
                        if config['frequency'] == '1h' and context['intraday_count'] != days * 4:
                            raise ValueError('Frozen hourly evaluation omitted registered intraday labels')
                        status['intraday_records'] = context['intraday_count']
                        status['status'] = 'complete'
                    else:
                        raise ValueError('Frozen rollout exceeded the requested horizon')
                context['generator'].close()
                atomic_json(context['output'] / 'status.json', status)
                local.append(status)
            phase(finish_case)
        finally:
            if active and 'generator' in context:
                try:
                    context['generator'].close()
                except Exception:
                    # Normal cleanup is inside finish_case's collective phase.
                    # Here preserve the shared exception already propagating.
                    pass
    def verify_frozen():
        current = model.state_dict()
        if current.keys() != frozen.keys() or any(not torch.equal(current[key].detach().cpu(), value)
                                                 for key, value in frozen.items()):
            raise RuntimeError('Frozen evaluation modified model weights or buffers')
    phase(verify_frozen)
    if distributed():
        groups = [None] * world
        dist.all_gather_object(groups, local)
        statuses = [case for group in groups for case in group]
    else:
        statuses = local
    statuses.sort(key=lambda case: (case['member'], case['start_day']))
    def summarize_and_render():
        if rank != 0:
            return
        report = summarize_cases(statuses, days)
        report.update(weight_metadata)
        atomic_json(root / 'summary.json', report)
        step = slots * days
        logger.log({'evaluation/completion_fraction': report['completion_fraction'],
                    'evaluation/completed_cases': report['completed_cases']}, step)
        from .visualization import render_evaluation
        for image in render_evaluation(root):
            logger.image(image, f'figures/{image.stem}', step)
        logger.artifact(root / 'summary.json', f'{stage}-{days}d-summary')
        logger.artifact(root / 'manifest.json', f'{stage}-{days}d-manifest')
        if report['completed_cases'] == len(cases):
            atomic_json(root / 'COMPLETE.json', {'days': days, 'cases': len(cases), 'stage': stage})
    phase(summarize_and_render)
    return root


def _paired_aggregate(pairs):
    """Equal case/lead weights on already spatially weighted squared errors."""
    if not pairs:
        raise ValueError('Paired aggregation requires at least one case/lead')
    model_rmse, physics_rmse, ratios = {}, {}, {}
    for field in FIELDS:
        key = f'post/{field}'
        model_mse = float(np.mean([pair['model_mse'][key] for pair in pairs]))
        baseline_mse = float(np.mean([pair['physics_mse'][key] for pair in pairs]))
        model_rmse[key] = math.sqrt(model_mse)
        physics_rmse[key] = math.sqrt(baseline_mse)
        ratios[key] = math.sqrt(model_mse / baseline_mse) if baseline_mse > 0 else None
    return {'paired_case_leads': len(pairs), 'model_rmse': model_rmse,
            'physics_rmse': physics_rmse, 'rmse_ratio': ratios}


def _paired_periods(pairs, cases, days):
    periods = {}
    for name, lower, upper in (('first50', 1, min(50, days)), ('whole', 1, days),
                               ('last50', max(1, days - 49), days)):
        selected = [pair for pair in pairs if lower <= pair['lead_day'] <= upper]
        expected_count = len(cases) * (upper - lower + 1)
        if len(selected) != expected_count:
            raise ValueError('Paired period does not cover every registered case and lead')
        per_case = []
        for member, start in cases:
            rows = [pair for pair in selected if pair['member'] == member and pair['start_day'] == start]
            per_case.append({'member': member, 'start_day': start, **_paired_aggregate(rows)})
        periods[name] = {'lead_day_start': lower, 'lead_day_end': upper, 'case_count': len(cases),
                         **_paired_aggregate(selected), 'per_case': per_case}
    return periods


def compare_to_physics(model_root, physics_root, output):
    """Pair every origin/lead explicitly; never replace missing baseline cases."""
    model_root, physics_root = Path(model_root), Path(physics_root)
    a = json.loads((model_root / "manifest.json").read_text())
    b = json.loads((physics_root / "manifest.json").read_text())
    if a["contract"] != b["contract"] or a["cases"] != b["cases"] or a["days"] != b["days"] or b["stage"] != "physics":
        raise ValueError("Model and pure-physics evaluation cases/contracts differ")
    rows, pairs = [], []
    if not a["cases"] or len({tuple(case) for case in a["cases"]}) != len(a["cases"]):
        raise ValueError("Paired comparison requires a nonempty unique case list")
    if not isinstance(a["days"], int) or a["days"] <= 0:
        raise ValueError("Paired comparison requires a positive integer horizon")
    for member, start in a["cases"]:
        name = f"m{member}_day{start:04d}"
        model = [json.loads(line) for line in (model_root / name / "daily.jsonl").read_text().splitlines()]
        physical = [json.loads(line) for line in (physics_root / name / "daily.jsonl").read_text().splitlines()]
        expected = list(range(1, a["days"] + 1))
        if [x["lead_day"] for x in model] != expected or [x["lead_day"] for x in physical] != expected:
            raise ValueError("Paired skill comparison requires complete lead sequences")
        for prediction, baseline in zip(model, physical):
            expected_time = start + prediction["lead_day"]
            _check_time(prediction['time'], expected_time)
            _check_time(baseline['time'], expected_time)
            if (abs(prediction["time"] - expected_time) > NOMINAL_CLOCK_TOL_DAYS or
                    abs(baseline["time"] - expected_time) > NOMINAL_CLOCK_TOL_DAYS or
                    abs(prediction["time"] - baseline["time"]) > NOMINAL_CLOCK_TOL_DAYS):
                raise ValueError("Paired evaluation model times do not match")
            ratios, model_mse, physics_mse = {}, {}, {}
            for field in FIELDS:
                numerator = prediction["metrics"][f"post/mse/{field}"]
                denominator = baseline["metrics"][f"post/mse/{field}"]
                if not all(math.isfinite(value) and value >= 0 for value in (numerator, denominator)):
                    raise ValueError('Paired MSE values must be finite and nonnegative')
                model_mse[f"post/{field}"] = float(numerator)
                physics_mse[f"post/{field}"] = float(denominator)
                ratios[f"post/{field}"] = float(np.sqrt(numerator / denominator)) if denominator > 0 else None
            pairs.append({"member": member, "start_day": start, "lead_day": prediction["lead_day"],
                          "model_mse": model_mse, "physics_mse": physics_mse})
            rows.append({"member": member, "start_day": start, "lead_day": prediction["lead_day"], "rmse_ratio": ratios})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, {"model_stage": a["stage"], "evaluation_field": "post_correction",
                         **{key: a.get(key) for key in ('selection', 'step', 'validated_windows', 'supervision', 'posttrain_gradient_mode')},
                         "days": a["days"], "cases": a["cases"], "daily": rows,
                         "aggregation": "equal weight per registered case and lead; square root after mean MSE; ratio of aggregate RMSE",
                         "zero_baseline_ratio": None,
                         "periods": _paired_periods(pairs, a["cases"], a["days"])})
    return output

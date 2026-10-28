"""Immutable evaluation protocol, independent of training split metadata."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

FIELDS = ('u', 'v', 't', 's', 'el')
STAGES = ('pretrain', 'midtrain4', 'posttrain', 'physics')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    import os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_protocol(path):
    value = read_json(path)
    if value.get('schema') != 'neuralpom-evaluation-v2':
        raise ValueError('Unsupported evaluation protocol')
    if value.get('inference') != 'bare_eager':
        raise ValueError('This campaign requires bare eager inference')
    if value.get('correction_policies') not in (['raw', 'constrained'], ['raw', 'constrained', 'density75']):
        raise ValueError('Both raw and historically constrained correction arms are required')
    horizons = value['horizons_days']
    if not horizons or horizons != sorted(set(horizons)) or any(type(d) is not int or d < 1 for d in horizons):
        raise ValueError('Horizons must be unique, positive, ordered whole days')
    cases = value['cases']
    if not cases or len({(c['member'], c['start_day']) for c in cases}) != len(cases):
        raise ValueError('Initial conditions must be nonempty and unique')
    for case in cases:
        if (type(case['member']) is not int or case['member'] != 7 or
                type(case['start_day']) is not int or case['start_day'] < 8000 or
                case['start_day'] + max(horizons) > 9000):
            raise ValueError('All labels must remain within member 7, days 8000–9000')
    groups = value['groups']
    if {(g['core'], g['frequency']) for g in groups} != {
            (c, f) for c in ('implicit_p0_fixed', 'explicit') for f in ('1h', '1day')} or len(groups) != 4:
        raise ValueError('Campaign needs all four core/frequency combinations')
    if len({g['id'] for g in groups}) != 4:
        raise ValueError('Duplicate group IDs')
    if 'reuse' in value:
        reuse=value['reuse']
        parent=read_json(Path(reuse['root'])/'campaign.json')
        if parent['protocol_sha256'] != reuse['protocol_sha256']:
            raise ValueError('Parent campaign identity changed')
        for key in ('cases','groups','horizons_days','inference'):
            if parent['protocol'][key] != value[key]:
                raise ValueError(f'Parent results are not scientifically paired: {key}')
        if reuse['policies'] != ['raw','constrained','physics']:
            raise ValueError('Only the original paired controls may be reused')
    seen = set()
    for entry in value.get("ai_baselines", []):
        if entry["id"] in seen or entry["id"] in {g["id"] for g in groups}:
            raise ValueError("Duplicate AI baseline ID")
        seen.add(entry["id"])
        if entry["core"] not in ("implicit_p0_fixed", "explicit") or entry.get("stages", ["pretrain", "posttrain"]) != ["pretrain", "posttrain"]:
            raise ValueError("AI baselines require a known reference core and two stages")
        if any(c["start_day"] <= hold <= c["start_day"]+max(horizons) for c in cases for hold in (8800, 8900)):
            raise ValueError("AI campaign touches reserved holdout days")
        from neuralpom.ai_baselines.config import load_config as load_ai_config
        cfg = load_ai_config(entry["config"])
        if cfg["core"] != entry["core"]:
            raise ValueError("AI reference core differs from registry")
    return value


def variants(protocol):
    for group in protocol['groups']:
        for stage in STAGES:
            for policy in (['physics'] if stage == 'physics' else protocol['correction_policies']):
                yield dict(group, stage=stage, policy=policy,
                           id=f"{group['id']}_{stage}_{policy}", group_id=group['id'])
    for entry in protocol.get("ai_baselines", []):
        group = next(g for g in protocol["groups"] if g["core"] == entry["core"] and g["frequency"] == "1day")
        for stage in ("pretrain", "posttrain"):
            yield dict(core=entry["core"], frequency="1day", config=entry["config"],
                model_family="ai", stage=stage, policy="ai_based",
                id=f"{entry['id']}_{stage}_ai_based", group_id=group["id"])


def case_id(case):
    return f"m{case['member']}_day{case['start_day']:04d}"


def case_root(root, variant_id, case):
    return Path(root) / 'runs' / variant_id / case_id(case)


def source_hashes(source):
    return {str(p.relative_to(source)): sha256(p) for p in sorted(Path(source).rglob('*.py'))
            if '__pycache__' not in p.parts}

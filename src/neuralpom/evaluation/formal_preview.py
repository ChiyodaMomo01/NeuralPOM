"""Publish a verified, dated formal-test preview without altering main statistics."""
from pathlib import Path
import csv
import hashlib
import html
import json
import math
import shutil


def append_preview(destination, base):
    registries = sorted(Path(base).glob('formal_comparison_preview_*/publication.json'))
    if not registries:
        return '', None, {}
    registry = registries[-1]
    source = registry.parent
    publication = json.loads(registry.read_text())
    if publication['schema'] != 'neuralpom-formal-preview-v1':
        raise ValueError('Unknown formal preview publication')
    hashes = publication['files']
    required = {'evidence.json','daily_mse_long.csv','common_metrics_audit.json','saved_inputs_audit.json',
                'mse_13_slots_preview10.png','mse_13_slots_preview10.svg','mse_13_slots_preview10.pdf'}
    if set(hashes) != required:
        raise ValueError('Incomplete formal preview bundle')
    for name, expected in hashes.items():
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Invalid preview filename')
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Formal preview artifact changed: ' + name)
    evidence = json.loads((source/'evidence.json').read_text())
    if (evidence['available_models'], evidence['requested_slots'], evidence['paired_origins'], evidence['days']) != (11,13,16,10):
        raise ValueError('Unexpected provisional comparison scope')
    rows = list(csv.DictReader((source/'daily_mse_long.csv').open()))
    expected_names = {f'implicit_p0_fixed_1h_{s}_{p}' for s in ('pretrain','midtrain4','posttrain')
                      for p in ('raw','constrained','density75')}
    expected_names.update(('implicit_p0_fixed_1h_physics_physics','implicit_p0_fixed_1h_physics_B1'))
    expected = {(name,7,origin,day,field) for name in expected_names
                for origin in range(8000,8481,32) for day in range(1,11)
                for field in ('u','v','t','s','el')}
    actual = [(r['variant'],int(r['member']),int(r['start_day']),int(r['lead_day']),r['field']) for r in rows]
    if len(actual)!=len(expected) or set(actual)!=expected:
        raise ValueError('Preview has missing, repeated, or unmatched case-days')
    if not all(math.isfinite(float(r['mse'])) and float(r['mse'])>=0 for r in rows):
        raise ValueError('Invalid preview metric')
    target = Path(destination)/'formal_preview'
    target.mkdir(exist_ok=True)
    for name in hashes:
        shutil.copy2(source/name,target/name)
    shutil.copy2(registry,target/'publication.json')
    artifacts = {'formal_preview/'+name:digest for name,digest in hashes.items()}
    artifacts['formal_preview/publication.json']=hashlib.sha256(registry.read_bytes()).hexdigest()
    metadata = dict(source=str(source),captured_at_utc=evidence['captured_at_utc'],
                    publication_sha256=artifacts['formal_preview/publication.json'],
                    available_models=11,requested_slots=13,paired_origins=16,days=10)
    markup = ('<section id="formal-test-preview"><h3>正式物理基线同图：固定十天初步预览</h3>'
        '<p><strong>Implicit 小时模型；16个共同起点、前10天，九条混合模型与B0/B1。</strong>'
        '物理基线用虚线，混合模型用实线；B2/B3尚无正式测试结果，仅保留待完成图例。'
        '校准数据未进入本图。本图不是50/200/300天最终结论。</p>'
        '<p>冻结预览时间：'+html.escape(evidence['captured_at_utc'])+'。各曲线使用各自已记录的物理单位MSE；'
        'B1运行于CPU，原对照运行于CUDA，网格存在舍入差异。全16起点前10天的1600项原始评分已复算；'
        '只改变评分面积权重的最大相对影响为1.51×10⁻⁸。这不限制长期预测差异，也不证明跨频率复用等价。</p>'
        '<img src="formal_preview/mse_13_slots_preview10.png" alt="固定16起点的五变量MSE；九混合模型与B0/B1；B2/B3待完成">'
        '<p><a href="formal_preview/daily_mse_long.csv">8800项逐初态指标CSV</a> · '
        '<a href="formal_preview/evidence.json">配对契约证据</a> · '
        '<a href="formal_preview/common_metrics_audit.json">全16起点评分复算</a> · '
        '<a href="formal_preview/saved_inputs_audit.json">初始与参考场审计</a> · '
        '<a href="formal_preview/mse_13_slots_preview10.pdf">PDF图</a></p></section>')
    return markup,metadata,artifacts

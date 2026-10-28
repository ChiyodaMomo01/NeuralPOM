"""Formal-test progress only; no calibration scores enter test rankings."""
from collections import Counter
from pathlib import Path
import hashlib
import html
import importlib.util
import json


def collect_formal_progress(base):
    output=[]
    for core,short in [('explicit','explicit'),('implicit_p0_fixed','implicit')]:
        for family in ('B1','B2','B3'):
            root=Path(base)/f'{family.lower()}_{short}_test_v1'
            path=root/'protocol.json'
            entry=dict(core=core,family=family,evaluation_split='test',requested=16,state='not_registered')
            output.append(entry)
            if not path.exists():continue
            protocol=json.loads(path.read_text());launcher=root/'physics_frozen_test.py'
            if hashlib.sha256(launcher.read_bytes()).hexdigest()!=protocol['launcher_sha256']:
                raise ValueError('Formal launcher differs from registration')
            spec=importlib.util.spec_from_file_location('_formal_progress_'+short+family,launcher)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            # This reader validates frozen source/selection/manifest and daily
            # commits without constructing a numerical runtime.
            data=module.collect_records(path)
            if data['variant']['core']!=core or data['variant']['baseline_family']!=family or len(data['cases'])!=16:
                raise ValueError('Formal family/origin contract differs')
            geometry_evidence={}
            for kind in ('geometry_difference','geometry_device_formula','geometry_weight_sensitivity'):
                paths=sorted(root.glob(kind+'_*.json'))
                if paths:
                    evidence=paths[-1]
                    geometry_evidence[kind]=dict(path=str(evidence),sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),data=json.loads(evidence.read_text()))
            entry['geometry_evidence']=geometry_evidence
            statuses=[c['status'] for c in data['cases']];counts=Counter(c['status'] for c in statuses)
            schedules=[];live=set();process_issues=[]
            for file in sorted((root/'results').glob('scheduler*.json')):
                s=json.loads(file.read_text());phase=s['days']
                for item in [s,*s['workers']]:
                    proc=Path(f"/proc/{item['pid']}/cmdline")
                    try:cmd=proc.read_bytes().decode().split('\0')
                    except (FileNotFoundError,ProcessLookupError,PermissionError):cmd=[]
                    item['live_matches']=str(launcher) in cmd and str(path) in cmd
                    item['actual_command']=' '.join(cmd)
                for worker in s['workers']:
                    if worker['live_matches']:live.add(worker['pid'])
                    elif worker['returncode'] is None:process_issues.append(dict(pid=worker['pid'],days=phase,issue='Recorded active worker is absent or command differs'))
                schedules.append(s)
            entry.update(state='registered',protocol_sha256=data['protocol_sha256'],selection_sha256=data['selection_sha256'],
                         parameters=protocol['parameters'],case_days=sum(s['completed_days'] for s in statuses),
                         status_counts=dict(counts),windows={str(h):dict(covered=sum(s['completed_days']>=h for s in statuses),requested=16) for h in (10,50,200,300)},
                         live_workers=len(live),worker_errors=data['worker_errors'],process_issues=list({(i['pid'],i['days']):i for i in process_issues}.values()),
                         schedulers=schedules,case_statuses=[dict(case=c['case'],status=c['status']) for c in data['cases']],
                         same_parameters_as_B0=family=='B1' and protocol['parameters']==dict(cm=.05,cc=.01))
    return output


def formal_section(campaigns):
    escape=lambda x:html.escape(str(x));rows=[];audit_notes=[]
    for c in campaigns:
        name=c['core']+' / '+c['family']
        if c['state']=='not_registered':
            rows.append(f'<tr><td>{escape(name)}</td><td colspan="7">尚未注册；仍保留16条目标</td></tr>');continue
        evidence=c.get('geometry_evidence',{})
        if all(k in evidence for k in ('geometry_difference','geometry_device_formula','geometry_weight_sensitivity')):
            g=evidence['geometry_difference']['data'];w=evidence['geometry_weight_sensitivity']['data']
            formula=evidence['geometry_device_formula']['data']
            if not all(all(v['formula_equals_saved'].values()) for v in formula['devices'].values()):
                raise ValueError('Grid formula evidence did not reproduce saved inputs')
            initial_same=all(a['equal_values'] and a['dtype']==a['B0_dtype'] for name in ('initial.npz','day001_reference.npz') for a in g['files'][name]['arrays'].values())
            input_note='初始场和首日参考场相同' if initial_same else '初始场或参考场不一致，禁止配对'
            forecast_note='首日预测场存在差异' if any(not a['equal_values'] for a in g['files']['day001_post.npz']['arrays'].values()) else '该保存首日预测场数值相同'
            dx=g['files']['geometry.npz']['arrays']['dx']['max_abs_difference']
            area=g['files']['geometry.npz']['arrays']['art']['relative_max']
            audit_notes.append('<p><strong>'+escape(name)+'：同参数不等于逐位相同轨迹。</strong> '
                +'原网格公式在CPU/CUDA上分别复现两套保存网格；该起点dx最大差 '+escape(f'{dx:.8g}')+' m，面积最大相对差 '+escape(f'{area:.5g}')
                +'。'+input_note+'，'+forecast_note+'。member '+escape(w['case']['member'])+' / start '+escape(w['case']['start_day'])+' 的前 '+escape(w['days'])
                +' 天，单独更换评分面积权重造成的MSE最大相对变化 '+escape(f"{w['max_abs_weight_only_relative_change']:.5g}")
                +'。这一短窗结果不证明16起点或300天等价，也不证明预测差异全部来自网格舍入；严格逐位复用条件尚未满足。'
                +'<a href="physics_calibration.json">数组、公式与权重敏感性证据及哈希</a>。</p>')
        counts=c['status_counts']
        state=f"{c['live_workers']} worker；数值失败 {counts.get('numerical_failure',0)}；系统错误 {counts.get('infrastructure_error',0)}；worker错误 {len(c['worker_errors'])}；进程待查 {len(c['process_issues'])}"
        note='参数与B0相同，保留对照，不宣称改进' if c['same_parameters_as_B0'] else '验证选参后冻结'
        values=[name,c['case_days'],*[f"{c['windows'][h]['covered']}/16" for h in ('10','50','200','300')],state,note]
        rows.append('<tr>'+''.join('<td>'+escape(x)+'</td>' for x in values)+'</tr>')
    return ('<section id="physics-formal-progress"><h2>物理基线：正式冻结测试进度</h2>'
            '<p>两核心三家族共96条目标，每家族16个起点，最长300天。此表是正式测试覆盖与运行状态，不是校准排名。尚未完成来源、跨频率和同初态配对核验的结果不自动加入效果比较；已查看测试集属于开发评估。</p>'
            '<div class="scroll"><table><thead><tr><th>核心 / 家族</th><th>轨迹日</th><th>10天</th><th>50天</th><th>200天</th><th>300天</th><th>进程与异常</th><th>参数解释</th></tr></thead><tbody>'
            +''.join(rows)+'</tbody></table></div>'+''.join(audit_notes)+'</section>')

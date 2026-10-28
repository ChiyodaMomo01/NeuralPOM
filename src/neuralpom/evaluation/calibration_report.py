"""Read-only validation progress appendix; never merges calibration into test metrics."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import json,hashlib,html


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def command(pid):
    p=Path(f'/proc/{pid}/cmdline');return p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else ''
def collect(base):
    base=Path(base);result=dict(captured_at_utc=datetime.now(timezone.utc).isoformat(),split='validation',campaigns=[])
    for core,short in [('explicit','explicit'),('implicit_p0_fixed','implicit')]:
        for family in ('B1','B2','B3'):
            root=base/f'{family.lower()}_{short}_screen_v1';path=root/'protocol.json'
            if not path.exists():result['campaigns'].append(dict(core=core,family=family,state='not_started'));continue
            p=read(path);assert p['core']==core and p['family']==family and p['selection_split']=='validation'
            source_hashes=read(root/'source_hashes.json')
            for name,h in source_hashes.items():assert digest(root/'source/src'/name)==h
            statuses={};counts=Counter();days=0
            for candidate in p['candidates']:
                for case in p['cases']:
                    file=Path(p['output_root'])/'runs'/candidate['id']/f"m{case['member']}_day{case['start_day']:04d}"/'status.json'
                    status=read(file) if file.exists() else dict(status='pending',completed_days=0)
                    statuses[(candidate['id'],case['member'],case['start_day'])]=status;counts[status['status']]+=1;days+=status['completed_days']
            schedulers=[]
            for file in sorted(Path(p['output_root']).glob('scheduler*.json')):
                scheduler=read(file)
                for item in [scheduler,*scheduler['workers']]:
                    cmd=command(item['pid']);item['actual_command']=cmd;item['live_matches']=str(path) in cmd and ('calibration' in cmd or 'continue_b1_validation' in cmd)
                schedulers.append(scheduler)
            cohorts={10:[c['id'] for c in p['candidates']]}
            for scheduler in schedulers:
                horizon=scheduler['days']
                if horizon==10:continue
                names=scheduler.get('candidates') or [w['candidate'] for w in scheduler['workers']]
                assert set(names)<={c['id'] for c in p['candidates']}
                cohorts[horizon]=list(dict.fromkeys(names))
            windows={}
            for horizon,names in cohorts.items():
                chosen=[statuses[(name,case['member'],case['start_day'])] for name in names for case in p['cases']]
                windows[str(horizon)]=dict(covered=sum(s['completed_days']>=horizon for s in chosen),requested=len(chosen))
            review_path=root/'selection50_final.json';review=read(review_path) if review_path.exists() else None
            if review is not None:assert review['protocol_sha256']==digest(path)
            screen_path=root/'selection10_final.json'
            screen=read(screen_path) if family in ('B2','B3') and screen_path.exists() else None
            if screen is not None:assert screen['protocol_sha256']==digest(path) and screen['case_support']==p['cases']
            workers=[w for s in schedulers for w in s['workers']]
            failures={k:counts[k] for k in ('numerical_failure','infrastructure_error')}
            result['campaigns'].append(dict(core=core,family=family,state='registered',protocol=p,protocol_sha256=digest(path),source_files_verified=len(source_hashes),status_counts=dict(counts),failures=failures,case_days=days,windows=windows,live_workers=sum(w['live_matches'] for w in workers),worker_errors=[w for w in workers if w['returncode'] not in (None,0)],schedulers=schedulers,review50=review,review10=screen))
    from .formal_progress import collect_formal_progress
    result["formal_tests"]=collect_formal_progress(base)
    return result


def section(data):
    esc=lambda x:html.escape(str(x));rows=[];details=[]
    for c in data['campaigns']:
        name=f"{c['core']} / {c['family']}"
        if c['state']=='not_started':rows.append(f'<tr><td>{esc(name)}</td><td colspan="6">尚未启动校准</td></tr>');continue
        windows=[f"{c['windows'][h]['covered']}/{c['windows'][h]['requested']}" if h in c['windows'] else '尚未启动' for h in ('10','50','200')]
        errors=c['failures'];state=f"{c['live_workers']} worker 运行；数值失败 {errors['numerical_failure']}，系统错误 {errors['infrastructure_error']}，worker错误 {len(c['worker_errors'])}"
        rows.append('<tr>'+''.join(f'<td>{esc(x)}</td>' for x in [name,c['case_days'],*windows,state,c['source_files_verified']])+'</tr>')
        screen=c.get('review10')
        if screen and screen['selection_ready']:
            ranked={r['candidate']['id']:r for r in screen['ranking']};selection=screen['selected_for_next_horizon'];table=[]
            for candidate in c['protocol']['candidates']:
                row=ranked.get(candidate['id']);score=f"{row['score']:.6f}" if row else '不可排名'
                status='进入50天' if candidate['id'] in selection else '保留记录'
                table.append('<tr>'+''.join(f'<td>{esc(x)}</td>' for x in [candidate['gamma'],candidate['beta'],score,status])+'</tr>')
            details.append(f'<h3>{esc(name)}：完整10天初筛</h3><p>四个验证起点完整窗口；按五变量MSE/B0比值均值排序，稳定候选前两名晋级。50/200天继续报告单变量10%与谱不劣化诊断，不以这些性能标准取消该家族；十天排名仅决定晋级。所有候选的预算、状态与逐变量误差保留在下载JSON。</p><div class="scroll"><table><thead><tr><th>gamma</th><th>beta</th><th>五变量平均MSE比值/B0</th><th>下一阶段</th></tr></thead><tbody>'+''.join(table)+'</tbody></table></div>')
        review=c['review50']
        if review and review['selection_ready']:
            ranked=[]
            for item in review['ranking']:
                candidate=item['candidate'];spectrum=review['spectral_reviews'][candidate['id']];q=spectrum.get('pooled',{}).get('depth_weighted',{}).get('candidate',{})
                number=lambda value: f'{value:.6f}' if value is not None else '不可定义'
                params=[candidate.get(k,c['protocol'].get(k,'—')) for k in ('cm','cc','gamma','beta')]
                ranked.append('<tr>'+''.join(f'<td>{esc(x)}</td>' for x in [*params,number(item['score']),number(q.get('energy_ratio')),number(q.get('shape_l1')),'通过' if spectrum['passes_spectral_gate'] else '未通过'])+'</tr>')
            selected=review.get('representative_for_200d') or review['recommended_for_200d']
            if review.get('selection_policy')=='within_family_validation_mse_v1' and selected is not None:
                interpretation='该家族按验证集MSE选择代表继续评估；相对B0的MSE与谱标准是诊断信息，不要求优于混合模型或B0。'
            elif selected is None:
                interpretation='本家族没有候选通过完整MSE与谱门槛，保留B0作为通用推荐；失败候选结果仍保留。'
            elif c['family']=='B1' and selected==review['baseline'] and any(x['score']<1 for x in review['ranking']):
                interpretation='低MSE候选未通过谱门槛，按固定规则保留原系数；未获得兼顾误差与谱的通用改进。'
            else:
                interpretation='按五变量与谱门槛选择，200天复核仍待完成。'
            selected=selected or '无合格候选'
            details.append(f'<h3>{esc(name)}：完整50天审查</h3><p>{esc(interpretation)} 进入200天复核的配置：{esc(selected)}。</p><div class="scroll"><table><thead><tr><th>Cm</th><th>Cc</th><th>gamma</th><th>beta</th><th>五变量平均MSE比值/B0</th><th>深度加权谱能量/参考</th><th>归一谱形状L1</th><th>完整谱门槛</th></tr></thead><tbody>'+''.join(ranked)+'</tbody></table></div>')
    return ('<section id="physics-calibration"><h2>物理基线：验证集校准进度</h2>'
        '<p><strong>B0–B3是公平、可复现的物理对照；不要求强于混合模型，也不因未优于B0而取消B2/B3家族。</strong> 各家族按验证集选择代表、冻结后完整测试；MSE、谱与能量预算用于如实分析优缺点。数值与预算错误仍须核验和披露。</p>'
        '<p><strong>校准进行中，尚未并入冻结测试效果排名。</strong> B1为系数校准，B2为四阶耗散，B3为同一耗散加能量预算回馈。四个验证起点为7000、7240、7480、7720；正式测试为另行冻结的16起点。下表分母随登记阶段候选数变化，不是同一组轨迹的完成率。</p>'
        f'<p>独立校准快照：{esc(data["captured_at_utc"])}。<a href="physics_calibration.json">下载完整校准快照、协议及审查证据</a>。</p>'
        '<div class="scroll"><table><thead><tr><th>核心 / 家族</th><th>已提交轨迹日</th><th>10天初筛</th><th>50天晋级</th><th>200天复核</th><th>进程与异常</th><th>源码文件核验</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'+''.join(details)+
        '<p>谱诊断同时检查深度加权及深度平均速度的能量幅度和谱形状；B1已完成的历史选择及门槛记录保留，后续B2/B3按家族代表推进；上表仅展示前一套诊断数值，完整结果在JSON。规则于四起点50天窗口完成前细化，不冒称十天初筛前预注册。低MSE、预算闭合或未失稳均不能单独证明物理统计正确；重叠测试初态也不作为独立样本。全部B1/B2/B3校准和长期测试完成后，才形成最终对照。</p></section>')


def comparison_brief(data, summary):
    """Put the directly comparable 50-day findings above the long report."""
    fields=('u','v','t','s','el')
    def pct(value):
        return '—' if value is None else f'{100*(value-1):+.1f}%'
    test_rows=[]
    for core,label in (('explicit','Explicit'),('implicit_p0_fixed','Implicit')):
        prefix=f'{core}_1h_'
        baseline=summary['groups'][prefix+'physics_physics']['periods']['50']
        reference=baseline['metrics']
        for suffix,method in (('physics_B1','B1 纯物理'),('posttrain_density75','Post-train density75 混合')):
            period=summary['groups'][prefix+suffix]['periods']['50']
            ratios=[pct(period['metrics'][f]['mse']/reference[f]['mse'])
                    if period['paired_cases']==16 and reference[f]['mse'] else '—' for f in fields]
            test_rows.append('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in
                [label,method,period['paired_cases'],*ratios])+'</tr>')
    validation_rows=[]
    for core,label in (('explicit','Explicit'),('implicit_p0_fixed','Implicit')):
        for family in ('B2','B3'):
            campaign=next(c for c in data['campaigns'] if c['core']==core and c['family']==family)
            review=campaign.get('review50')
            selected=review.get('representative_for_200d') if review else None
            ranked=next((r for r in review['ranking'] if r['candidate']['id']==selected),None) if selected else None
            ratios=[pct(ranked['ratios'][f]) if ranked else '—' for f in fields]
            validation_rows.append('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in
                [label,family,4 if ranked else 0,*ratios])+'</tr>')
    headings='<thead><tr><th>核心</th><th>方法</th><th>初态 n</th><th>U</th><th>V</th><th>T</th><th>S</th><th>SSH</th></tr></thead>'
    return ('<section id="comparison-at-a-glance"><h2>先看结论 / 物理基线与混合模型</h2>'
        '<p class="note"><strong>读法：</strong>下表是逐变量 MSE 相对同核心 B0 的变化；负值表示误差降低，正值表示增加。'
        '五个变量各有物理单位，不能把百分比相加当作总误差。B2/B3 的验证集与下方冻结测试不是同一组初态，不据此推断其正式测试优劣。</p>'
        '<h3>冻结测试 · 相同 16 个初态 · 完整第 1–50 天</h3>'
        '<p>B0 是原纯物理基线。B1 采用与 B0 相同的历史参数，运行结果仅有微小数值差异。'
        '混合模型展示 Post-train + density75 的逐小时分支，不代表全部九个混合分支。</p>'
        '<div class="scroll"><table>'+headings+'<tbody>'+''.join(test_rows)+'</tbody></table></div>'
        '<h3>验证集 · 相同 4 个起点 · 完整第 1–50 天</h3>'
        '<p>B2/B3 展示各家族按验证集规则选出的代表相对 B0 的结果；即使误差增加，仍保留为公平对照。'
        '正式测试尚未形成相应的完整 50 天配对时，不能声称 B2/B3 相对混合模型已有测试结论。</p>'
        '<div class="scroll"><table>'+headings+'<tbody>'+''.join(validation_rows)+'</tbody></table></div>'
        '<p><a href="physics_calibration.json">验证集候选与审查证据</a> · '
        '<a href="formal_comparison/summary.json">冻结测试逐变量汇总</a> · '
        '<a href="#physics-calibration">校准细节</a></p></section>')


def append_to_report(directory,base):
    directory=Path(directory);data=collect(base);payload=section(data)
    from .formal_progress import formal_section
    payload+=formal_section(data["formal_tests"])
    from .formal_preview import append_preview
    preview,metadata,artifacts=append_preview(directory,base)
    payload+=preview
    data['formal_preview']=metadata
    from .protocol import load_protocol
    from .formal_report import export_comparison
    comparison_protocol=Path(base).parent/'evaluation_12checkpoints_20260927_optimized/source/configs/evaluation/paired_optimized_20260927.json'
    comparison,comparison_summary,comparison_artifacts=export_comparison(load_protocol(comparison_protocol),base,directory)
    from .formal_verification import verify_comparison
    verification=verify_comparison(directory,load_protocol(comparison_protocol))
    payload+=comparison
    artifacts.update(comparison_artifacts)
    data['formal_comparison']={'captured_at_utc':comparison_summary['captured_at_utc'],
        'summary_path':'formal_comparison/summary.json',
        'csv_verification':verification,
        'scope_note':comparison_summary['scope_note']}
    index=directory/'index.html';text=index.read_text();marker='<h2>05 / 来源与复现</h2>'
    if text.count(marker)!=1 or 'id="physics-calibration"' in text:raise ValueError('Unexpected report structure')
    (directory/'physics_calibration.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    text=text.replace(marker,payload+marker)
    first='<h3>推进覆盖度</h3>'
    if text.count(first)!=1:raise ValueError('Missing overview insertion point')
    index.write_text(text.replace(first,comparison_brief(data,comparison_summary)+first,1))
    build=read(directory/'build.json');build['analysis_sources']['calibration_report.py']=digest(__file__)
    build['analysis_sources']['formal_progress.py']=digest(Path(__file__).with_name('formal_progress.py'))
    build['analysis_sources']['formal_preview.py']=digest(Path(__file__).with_name('formal_preview.py'))
    for name in ('formal_report.py','formal_registry.py','formal_verification.py','comparison_interpretation.py','physics_comparison.py','formal_budgets.py','budget_windows.py','plots.py'):
        build['analysis_sources'][name]=digest(Path(__file__).with_name(name))
    build['artifacts'].update(artifacts)
    for name in ('index.html','physics_calibration.json'):build['artifacts'][name]=digest(directory/name)
    (directory/'build.json').write_text(json.dumps(build,indent=2)+'\n')
    return data


def progress_signature(data):
    """Ignore polling timestamps, but retain scientifically relevant progress."""
    keys=('core','family','state','protocol_sha256','case_days','windows','status_counts','failures','live_workers','worker_errors','review50','review10')
    formal_keys=('core','family','state','protocol_sha256','selection_sha256','parameters','case_days','windows','status_counts','live_workers','worker_errors','process_issues','geometry_evidence')
    return json.dumps(dict(calibration=[{k:c.get(k) for k in keys} for c in data['campaigns']],formal=[{k:c.get(k) for k in formal_keys} for c in data.get('formal_tests',[])]),sort_keys=True)

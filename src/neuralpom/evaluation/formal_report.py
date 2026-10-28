"""Unified formal baseline artifacts; original campaign statistics stay separate."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import html
import json
from .formal_registry import collect_with_formal
from .physics_comparison import unified_summary
from .formal_budgets import collect_formal_budgets
from .plots import plot_mse, plot_field_comparison, plot_spectra, save


def export_comparison(protocol, base, destination, *, spatial=True):
    out=Path(destination)/'formal_comparison';out.mkdir(parents=True,exist_ok=True)
    data,roots,provenance=collect_with_formal(protocol,base)
    summary=unified_summary(protocol,data)
    summary['source_provenance']=provenance
    summary['captured_at_utc']=datetime.now(timezone.utc).isoformat()
    summary['display_slots']=len(data)*len(protocol['cases'])
    summary['unique_target_trajectories']=640+96
    summary['budgets']={name:collect_formal_budgets(data[name]) for name in roots}
    summary['scope_note']='Registered-method figures omit only explicit unavailable slots; fixed full-window origins include every registered failure. Strict all-13-method windows remain separate. Daily reuse stays pending. Display aliases are not independent trajectories.'
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    from .comparison_interpretation import export_interpretation, interpretation_section
    interpretation=export_interpretation(summary,out)
    fields=('u','v','t','s','el')
    with (out/'daily_mse.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['variant','member','start_day','lead_day',*[f'mse_{f}' for f in fields]])
        for name,model in data.items():
            for case in model['cases']:
                for row in case['rows']:
                    writer.writerow([name,case['case']['member'],case['case']['start_day'],row['lead_day'],*[row['metrics']['mse/'+f] for f in fields]])
    with (out/'paired_mse.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['group','baseline','model','window','paired_cases','field','model_mse','comparator_mse','relative_mse_change'])
        for gid,baselines in summary['physics_pairs'].items():
            for family,models in baselines.items():
                for name,windows in models.items():
                    for window,pair in windows.items():
                        for field,m in pair['metrics'].items():
                            writer.writerow([gid,family,name,window,pair['paired_cases'],field,m['model_mse'],m['comparator_mse'],m['relative_mse_change']])
    figures=[]
    for horizon in (10,50,200,300):
        figures.append(plot_mse(summary,out,mode='registered',horizon=horizon))
    if spatial:
        # Fixed first registered origin, fixed lead days; no selection by skill.
        case=protocol['cases'][0]
        for group in protocol['groups']:
            gid=group['id'];names=[n for n,m in data.items() if m['variant']['group_id']==gid]
            active=[n for n in names if data[n].get('availability') not in ('not_registered','reuse_pending')]
            common=min(data[n]['cases'][0]['status']['completed_days'] for n in active)
            for day in (50,200,300):
                if day>common:continue
                for field in ('el','u','v'):
                    for subtract,suffix in ((False,'absolute'),(True,'change')):
                        figures.append(save(plot_field_comparison(protocol,names,case,day,field=field,
                            subtract_initial=subtract,run_roots=roots),out/f'{gid}_{day}d_{field}_{suffix}'))
                for first,suffix in ((1,'full'),(max(1,day-49),'last50')):
                    figures.append(save(plot_spectra(protocol,names,case,first,day,run_roots=roots),out/f'{gid}_{day}d_spectrum_{suffix}'))
    rows=[]
    for name,m in data.items():
        if m['variant']['stage']!='physics':continue
        status=m.get('availability','original_B0')
        counts=[sum(c['status']['completed_days']>=h for c in m['cases']) for h in (50,200,300)]
        errors=sum(c['status']['status'] in ('numerical_failure','infrastructure_error') for c in m['cases'])
        rows.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in (name,status,*counts,errors))+'</tr>')
    links='<p><a href="formal_comparison/summary.json">完整来源、配对、覆盖、失败与预算JSON</a> · <a href="formal_comparison/daily_mse.csv">逐初态MSE</a> · <a href="formal_comparison/paired_mse.csv">B0–B3全部成对比较</a></p>'
    links+='<p><a href="formal_comparison/ANALYSIS.md">同初态定量分析与解释边界</a> · <a href="formal_comparison/fixed_cohort_analysis.json">完整窗口及末50天的固定初态分析JSON</a></p>'
    markup=('<section id="unified-formal-comparison"><h3>九混合分支与B0–B3统一比较（进行中）</h3>'
        '<p>物理虚线，混合实线；每个窗口固定同一组完整起点。未登记或复用待核验的分支保留图例，不能把暂缺方法当作已完成。'
        '已登记失败仍约束共同窗口，所有失败独立披露。'+html.escape(summary['captured_at_utc'])+'</p>'
        '<p>52×16是显示槽位，唯一目标为原640条加新增96条；日频别名不作为独立样本。'
        '正式物理基线CPU与原CUDA实现存在已披露网格舍入差异，评分采用各自记录权重。来源检查不替代全窗口评分复算；'
        '低MSE、预算闭合或未失稳均不证明物理统计正确。校准未进入本区。</p>'+links+interpretation_section(interpretation)+
        '<div class="scroll"><table><thead><tr><th>分支</th><th>登记/复用状态</th><th>50天/16</th><th>200天/16</th><th>300天/16</th><th>失败</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'+
        ''.join('<details><summary>'+html.escape(p.stem)+'</summary><img src="formal_comparison/'+html.escape(p.name)+'" alt="'+html.escape(p.stem)+'"></details>' for p in figures)+
        '<p>空间图为协议首起点、固定50/200/300天，EL/U/V原场与减同初始场，RdBu_r零对称显示范围；谱为相同起点完整与末50天诊断。预算按增量复算，B1无回馈账本不能解释为零能量测量。</p></section>')
    (out/'section_fragment.txt').write_text(markup)
    hashes={str(p.relative_to(Path(destination))):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    return markup,summary,hashes

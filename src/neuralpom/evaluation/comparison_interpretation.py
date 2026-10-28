"""Describe complete-cohort errors without pooling incompatible physical units."""
from pathlib import Path
import json
import math
import html

FIELDS=('u','v','t','s','el')
UNITS={'u':'m² s⁻²','v':'m² s⁻²','t':'°C²','s':'psu²','el':'m²'}


def interpretation_section(data):
    esc=lambda value:html.escape(str(value))
    sections=[]
    for group,windows in data['periods'].items():
        for key,period in windows.items():
            title=f"{group} / {key}：完整共同初态 {period['n']}，已登记方法 {len(period['included_variants'])}/13"
            content=f"<p>全部13方法的完整共同初态：{period['strict_all_method_n']}。数据时间：{esc(data['captured_at_utc'])}。</p>"
            if period['unavailable_variants']:
                content+='<p>待核验或待完成：'+esc(', '.join(period['unavailable_variants']))+'。</p>'
            if not period['n']:
                content+='<p>没有完整共同窗口，不报告排名或收益。</p>'
            else:
                content+='<div class="scroll"><table><thead><tr><th>方法</th>'+''.join(f'<th>{f.upper()} ({esc(UNITS[f])})</th>' for f in FIELDS)+'</tr></thead><tbody>'
                for name,values in period['mse'].items():
                    content+='<tr><td>'+esc(name)+'</td>'
                    for field in FIELDS:
                        relative=period['relative_to_B0'][name][field]
                        suffix='未定义' if relative is None else f'{relative*100:+.3f}%'
                        content+=f'<td>{values[field]:.6g}<br><small>相对B0：{suffix}</small></td>'
                    content+='</tr>'
                content+='</tbody></table></div>'
            sections.append('<details><summary>'+esc(title)+'</summary>'+content+'</details>')
    return ('<section id="fixed-cohort-interpretation"><h3>固定共同初态：逐变量定量比较</h3>'
        '<p>阶段性描述分析；每个窗口内方法共享同一组完整初态。负百分比表示该变量MSE低于B0；不同单位不合并为总分。'
        '末50天沿用对应完整窗口初态，跨窗口初态可能不同。缺失或失败不填零，完整窗口均值条件于可用完整轨迹。</p>'+
        ''.join(sections)+'<p>重叠初态并非独立样本；CPU/CUDA舍入差异、谱、空间场与预算须结合解释。'
        '当前误差或未失稳不能单独证明物理统计正确，也不构成不同核心的因果优劣证明。</p></section>')


def fixed_periods(summary):
    result={}
    for group,windows in summary['registered_curve_windows'].items():
        result[group]={}
        for horizon in (50,200,300):
            window=windows[str(horizon)]
            names=window['included_variants']
            for first,suffix in [(1,''),*([(horizon-49,'_last50')] if horizon>50 else [])]:
                models={}
                for name in names:
                    rows=window['daily'][name]
                    if not window['n']:
                        if rows:raise ValueError('Empty cohort has data')
                        continue
                    if [r['lead_day'] for r in rows]!=list(range(1,horizon+1)):raise ValueError('Incomplete fixed window')
                    if any(r['n']!=window['n'] for r in rows):raise ValueError('Changing cohort size')
                    values={f:sum(r['mse'][f] for r in rows[first-1:horizon])/(horizon-first+1) for f in FIELDS}
                    if not all(math.isfinite(v) and v>=0 for v in values.values()):raise ValueError('Invalid fixed-cohort MSE')
                    models[name]=values
                b0=models.get(group+'_physics_physics')
                ratios={name:{f:values[f]/b0[f]-1 if b0 and b0[f] else None for f in FIELDS} for name,values in models.items()}
                result[group][str(horizon)+suffix]=dict(first_day=first,last_day=horizon,n=window['n'],
                    cases=window['cases'],included_variants=names,unavailable_variants=window['unavailable_variants'],
                    strict_all_method_n=summary['curve_windows'][group][str(horizon)]['n'],
                    mse=models,relative_to_B0=ratios,conditional_on_complete_origins=True)
    return result


def export_interpretation(summary,destination):
    out=Path(destination);out.mkdir(parents=True,exist_ok=True)
    periods=fixed_periods(summary)
    data={'captured_at_utc':summary['captured_at_utc'],'units':UNITS,'periods':periods,
        'scope':'Fixed registered-method complete cohorts; unavailable methods explicit; no unit pooling or independent-origin significance claim.'}
    (out/'fixed_cohort_analysis.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    lines=['# 物理基线与混合模型：阶段性定量分析','',
        '**本文件是当前快照的描述性分析，完整研究尚未结束。** 数据时间：'+summary['captured_at_utc']+'。','',
        '## 比较口径','',
        '- 每个窗口固定同一组完整初态；同一组内所有已登记方法共享这些初态。未登记/复用待核验分支不能被当成已完成。',
        '- 各变量分别报告物理单位MSE；不同单位之间不求平均总分。相对B0变化小于0表示该变量误差较低。',
        '- 末50天与完整窗口使用相同的完整初态支持；不同窗口的初态集合可能不同，不能直接将跨窗口差值归因为误差增长。',
        '- 失败或短轨迹不能进入完整窗口平均；因此结果条件于完整窗口，需联合覆盖和失败表阅读，不能解释为无条件总体性能。',
        '- 重叠初态不作为独立样本；此处不报告独立样本显著性。跨核心数据不同，不作核心因果优劣证明。','',
        '## 已核验的数值比较','']
    header='| 方法 | '+' | '.join(f'{f.upper()} ({UNITS[f]})' for f in FIELDS)+' |'
    divider='|---|'+'---:|'*len(FIELDS)
    for group,windows in periods.items():
        for key,period in windows.items():
            lines+=['### '+group+' / '+key+' 天窗口','',
                f"完整共同初态 **{period['n']}**；已登记方法 **{len(period['included_variants'])}/13**；全部13方法完整共同初态 **{period['strict_all_method_n']}**。",'']
            if period['unavailable_variants']:
                lines+=['待提供结果：'+', '.join(f'{name} ({why})' for name,why in period['unavailable_variants'].items())+'。','']
            if not period['n']:
                lines+=['当前没有完整共同窗口，不给出排名或收益结论。',''];continue
            lines+=[header,divider]
            for name,values in period['mse'].items():lines+=['| '+name+' | '+' | '.join(f'{values[f]:.6g}' for f in FIELDS)+' |']
            lines+=['','相对同初态B0的MSE变化（%）：','', '| 方法 | U | V | T | S | EL |',divider]
            for name,values in period['relative_to_B0'].items():lines+=['| '+name+' | '+' | '.join('未定义' if values[f] is None else f'{100*values[f]:+.3f}%' for f in FIELDS)+' |']
            lines+=['']
    lines+=['## 分析边界与待解释问题','',
        '- B1历史选择可能与B0系数相同；CPU正式物理基线与原CUDA基线存在已披露的网格舍入差异。二者偏差不自动视为物理闭合收益。',
        '- 较低MSE、预算闭合或观察期内无数值失败，都不足以证明长期物理统计正确。须联合动能谱、空间结构、预算与限制器作用解释。',
        '- 当前表不单独评判Explicit是否更易崩溃；该判断须报告每个配置的样本范围、观察时长、首次失败及系统错误，且不得把未完成视为存活到300天。',
        '- 验证选参结果不混入测试表；已被研究者查看的测试属于开发评估，不能重新包装为从未查看的泛化证据。','']
    (out/'ANALYSIS.md').write_text('\n'.join(lines))
    return data

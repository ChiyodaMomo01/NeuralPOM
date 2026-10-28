"""A self-contained, source-linked HTML snapshot of the live evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
from pathlib import Path

from .protocol import load_protocol, atomic_json, read_json
from .analysis import export
from .plots import plot_mse, plot_coverage, plot_fields, plot_spectra, plot_field_comparison, save


def number(value):
    return '—' if value is None else f'{value:.5g}'


def progress_tables(summary, protocol):
    """Show completed observation windows, distinct from final trajectory status."""
    horizons=[10,*protocol['horizons_days']]
    rows=[]
    for group in protocol['groups']:
        items=[i for i in summary['groups'].values() if i['variant']['group_id']==group['id']]
        counts=[sum(next((d['n'] for d in i['daily'] if d['lead_day']==h),0) for i in items) for h in horizons]
        longest=max((d['lead_day'] for i in items for d in i['daily']),default=0)
        values=[group['id'],len(items)*summary['n_requested'],*counts,longest]
        rows.append('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in values)+'</tr>')
    head=['核心 / 频率','目标轨迹数',*[f'已覆盖 {h} 天' for h in horizons],'最长 lead / 天']
    return '<h3>推进覆盖度</h3><p>计数单位是“配置 × 初态”的轨迹；同一初态的不同策略分别计数，不能当作独立样本。覆盖 50 天不等于 300 天任务完成。</p><div class="scroll"><table><thead><tr>'+''.join(f'<th>{v}</th>' for v in head)+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'


def table_markup(headers, rows):
    return ('<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+
            '</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+
            '</tbody></table></div>')


def long_window_tables(summary, protocol):
    sections=[]
    fields=('u','v','t','s','el')
    labels=['U','V','T','S','SSH']
    def relative(metrics):
        return ['—' if metrics[f]['relative_mse_change'] is None else f"{100*metrics[f]['relative_mse_change']:+.1f}%" for f in fields]
    for horizon in protocol['horizons_days']:
        full=[];tail=[];policies=[];stages=[]
        for name,item in summary['groups'].items():
            p=item['periods'][str(horizon)]
            if p['paired_cases']:
                full.append([name,p['paired_cases'],*[number(p['metrics'][f]['mse']) for f in fields],*relative(p['metrics'])])
            p=item['periods'].get(f'{horizon}_last50')
            if p and p['paired_cases']:
                tail.append([name,p['paired_cases'],*[number(p['metrics'][f]['mse']) for f in fields],*relative(p['metrics'])])
        for name,periods in summary['optimized_pairs'].items():
            for label,p in periods[str(horizon)].items():
                if p['paired_cases']:
                    policies.append([name,label,p['paired_cases'],*relative(p['metrics'])])
        for name,periods in summary.get('stage_pairs',{}).items():
            p=periods[str(horizon)]
            if p['paired_cases']:stages.append([name,p['paired_cases'],*relative(p['metrics'])])
        count=len(full)
        body=(f'<p>第 1–{horizon} 天完整窗口；只有模型与比较对象共同覆盖的初态进入统计。正百分比表示误差更大。'
              '物理基线配对、策略配对和阶段配对的初态数可能不同；不按非配对行直接排序。</p>')
        headers=['配置','共同 n',*[x+' MSE' for x in labels],*[x+' vs Physics' for x in labels]]
        body+=table_markup(headers,full) if full else '<p>尚无与物理基线完整配对的结果。</p>'
        if tail:
            body+=f'<h4>末段：第 {horizon-49}–{horizon} 天</h4>'+table_markup(headers,tail)
        if policies:
            body+='<h4>同权重策略对照：raw_vs_optimized 为裸推理相对 density75；optimized_vs_standard 为 density75 相对基础约束</h4>'+table_markup(['权重','比较','共同 n',*labels],policies)
        if stages:
            body+='<h4>同策略阶段对照：Post-train 相对 Midtrain 的 MSE 变化</h4>'+table_markup(['核心 / 频率 / 策略','共同 n',*labels],stages)
        ai_rows=[]
        for name, periods in summary.get('ai_pairs', {}).items():
            pair=periods[str(horizon)]
            if pair['paired_cases']:
                ai_rows.append([name,pair['comparator'],pair['paired_cases'],*relative(pair['metrics'])])
        if ai_rows:
            body+='<h4>纯 AI 相对同核心每日 Post-D75：完整共同窗口</h4>'+table_markup(
                ['AI配置','比较对象','共同 n',*labels],ai_rows)
        sections.append(f'<details {"open" if count else ""}><summary><b>{horizon} 天：{count} 个配置已有完整配对结果</b></summary>'+body+'</details>')
    return '<h3>50 / 200 / 300 天与末段结果</h3>'+''.join(sections)


def optimized_notes(summary):
    pairs=[(name, p['10']) for name,p in summary['optimized_pairs'].items()
           if name.endswith('_posttrain') and p['10']['raw_vs_optimized']['paired_cases']]
    if not pairs:
        return '裸推理与 density75 尚无完整 10 天共同窗口，暂不作效果判断。'
    items=[]
    for name,pair in pairs:
        p=pair['raw_vs_optimized']; n=p['paired_cases']
        changes=[]
        for f,label in [('u','U'),('v','V'),('t','T'),('s','S'),('el','SSH')]:
            value=p['metrics'][f]['relative_mse_change']
            changes.append(f'{label} {100*value:+.1f}%' if value is not None else f'{label} —')
        label=name.replace('implicit_p0_fixed','Implicit').replace('explicit','Explicit').replace('_posttrain','').replace('_',' / ')
        items.append(f'<li><b>{label}</b>：裸推理相对 density75 的 MSE 变化为 '+
                     '、'.join(changes)+f'；共同初态 n={n}。</li>')
    return ('<b>已知事实 · 主对照：</b>前 10 天 Post-train 的完整成对结果如下，正值表示裸推理误差更高。'
            '<ul>'+''.join(items)+'</ul><b>分析：</b>这些结果仅比较同一冻结权重的推理策略。'
            '它们不能证明 Post-train 已充分收敛，也不能证明某核心在 50–300 天更稳定。'
            'density75 对 Explicit、日频及其他阶段的收益仍属迁移检验；应保留所有变量的改善与退化。')


def evidence_notes(summary):
    """State only comparisons with actual paired support, never infer causation."""
    pairs=[v['10'] for name,v in summary['correction_pairs'].items()
           if name.endswith('_posttrain') and v['10']['paired_cases'] > 0]
    ratios=[p['metrics']['t']['raw_relative_to_constrained']+1 for p in pairs
            if p['metrics']['t']['raw_relative_to_constrained'] is not None]
    if not ratios:
        return '目前还没有完整的前 10 天 Post-train 成对结果，暂不判断约束带来的误差变化。'
    counts=[p['paired_cases'] for p in pairs]
    text=(f'已知事实：前 10 天已有 {len(pairs)} 组 Post-train 权重可成对比较，每组共同初态数 '
          f'{min(counts)}–{max(counts)}。裸校正的温度 MSE 为约束版的 '
          f'{min(ratios):.2f}–{max(ratios):.2f} 倍（这里的约束版指基础约束，不是 density75）。')
    velocity=[p['metrics'][f]['raw_relative_to_constrained'] for p in pairs for f in ('u','v')]
    if all(x is not None and x>0 for x in velocity):
        text+='这些对照中，裸校正的 U、V MSE 也都更高。'
    ssh=[p['metrics']['el']['raw_relative_to_constrained'] for p in pairs]
    if any(x is not None and x<0 for x in ssh):
        text+='但部分 SSH MSE 在裸校正下更低，收益并非所有变量同向。'
    return text+' 分析：当前证据首先显示约束对短期温度预测的重要性；误差增长的来源与长期稳定性仍需更长窗口和更多初态验证。'


def render(protocol_path, destination):
    protocol=load_protocol(protocol_path);destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    summary=export(protocol,destination)
    figures=[plot_mse(summary,destination),plot_mse(summary,destination,horizon=10),
             plot_mse(summary,destination,mode='available'),plot_coverage(summary,destination)]
    # Each family receives a representative field/spectrum at the latest shared
    # committed day of its first origin. No cherry-picking by error magnitude.
    for group in protocol['groups']:
        names=[name for name,item in summary['groups'].items() if item['variant']['group_id']==group['id']]
        case=protocol['cases'][0]
        statuses=[]
        for name in names:
            path=Path(protocol['output_root'])/'runs'/name/f"m7_day{case['start_day']:04d}"/'status.json'
            statuses.append(read_json(path)['completed_days'] if path.exists() else 0)
        common=min(statuses)
        if common:
            figures.append(save(plot_spectra(protocol,names,case,1,common),destination/f"{group['id']}_spectra"))
            if common>50:
                figures.append(save(plot_spectra(protocol,names,case,common-49,common),destination/f"{group['id']}_last50_spectra"))
        # Fixed, declared dates and the first registered origin; never choose a
        # field snapshot by apparent skill. All methods share one color scale.
        for lead in (50,200,300):
            if lead>common:continue
            for field in ('el','u','v'):
                for subtract_initial,suffix in ((False,'absolute'),(True,'change_from_initial')):
                    figures.append(save(plot_field_comparison(protocol,names,case,lead,field=field,
                        subtract_initial=subtract_initial),
                        destination/f"{group['id']}_all_methods_day{lead:03d}_{field}_{suffix}"))
        for policy in protocol['correction_policies']:
            name=f"{group['id']}_posttrain_{policy}";days=statuses[names.index(name)]
            if days:
                for subtract_initial,suffix in ((False,'fields'),(True,'change_from_initial')):
                    figures.append(save(plot_fields(protocol,name,case,days,field=('el','u','v'),
                        subtract_initial=subtract_initial,cmap='RdBu_r',symmetric_clim=True),
                        destination/f"{group['id']}_{policy}_el_u_v_{suffix}"))
    audit_source=Path(protocol['output_root'])/'curve_audit_20260927T0715'
    audit_note=''
    if (audit_source/'index.html').exists():
        import shutil
        shutil.copytree(audit_source,destination/'curve_audit',dirs_exist_ok=True)
        audit_note='<section class="note"><h3>误差曲线突降审计（15:15 冻结快照）</h3><p><b>第10天后的32次下降均来自参与平均的初态减少；固定同初态后，这32处均继续上升。</b>例如Explicit日频物理基线V在42→43天显示−5.74%，来自n=4→3。早期EL回落则在固定16初态中也存在，不能一起归为换样本。已核对75800项原始日志指标，并独立重算1110项保存场MSE。主图现固定共同初态，辅助图在换样本处断线。</p><p><a href="curve_audit/index.html">查看完整审计、分解图与原始证据</a> · <a href="curve_audit/jump_decomposition.csv">9800项变化分解CSV</a></p></section>'
    rows=[];failures=[];exposures=[]
    for name,item in summary['groups'].items():
        v=item['variant'];s=item['statuses'];period=item['periods']['preview10']
        started=sum(s.values())-s['pending'];exposures.append(item['completed_case_days'])
        rows.append('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in [
            v['core'].replace('_p0_fixed',''),v['frequency'],v['stage'],v['policy'],
            f"{s['complete']}/{summary['n_requested']}",f"{s['numerical_failure']}/{started}",period['paired_cases'],
            *[number(period['metrics'][f]['mse']) for f in ('u','v','t','s','el')]])+'</tr>')
        for failure in item['failures']:
            failures.append(f"<li><b>{html.escape(name)}</b> · start {failure['start_day']} · {failure['status']} · day {failure.get('first_anomaly_day')} · {html.escape(str(failure.get('error')))}</li>")
    total_failed=sum(i['statuses']['numerical_failure'] for i in summary['groups'].values())
    pending=sum(i['statuses']['pending']+i['statuses']['running'] for i in summary['groups'].values())
    infrastructure=sum(i['statuses']['infrastructure_error'] for i in summary['groups'].values())
    for error in summary['worker_errors']:
        failures.append(f"<li>Worker error: {html.escape(error['variant'])}, exit {error['returncode']}; inspect its log. This is not counted as numerical instability.</li>")
    pair_rows=[]
    for name,periods in summary['correction_pairs'].items():
        p=periods['10']
        values=[]
        for field in ('u','v','t','s','el'):
            v=p['metrics'][field]['raw_relative_to_constrained']
            values.append('—' if v is None else f'{100*v:+.1f}%')
        pair_rows.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [name,p['paired_cases'],*values])+'</tr>')
    pair_table='<h3>同权重：Raw 相对 Constrained 的前 10 天 MSE 变化</h3><p>正值表示裸校正误差更大。每一行只使用两策略共同完整覆盖的相同初态；n = 0 时不作比较。</p><div class="scroll"><table><thead><tr>'+''.join('<th>'+x+'</th>' for x in ['权重','共同 n','U','V','T','S','SSH'])+'</tr></thead><tbody>'+''.join(pair_rows)+'</tbody></table></div>'
    conclusion=('目前尚未观察到数值失稳；有限的观察期不能证明长期稳定，也暂不支持“Explicit 更容易崩溃”。'
        if not total_failed else '已观察到数值失稳。应按核心、频率、阶段与约束策略分层比较，不能把所有失败归因于 Explicit 核心。')
    now=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    cards=''.join(f'<div class="card"><strong>{n}</strong><span>{label}</span></div>' for n,label in [
        (12,'冻结模型'),(12*len(protocol['correction_policies']),'模型推理分支'),(summary['n_requested'],'初始条件 / 配置'),(sum(exposures),'已提交轨迹天'),(total_failed,'数值失败轨迹')])
    def figure_label(p):
        if '_all_methods_' in p.stem:return '同初态/同日期全部已登记方法：共享零对称色标；单起点空间示例，不替代16初态统计 · '+p.stem
        if p.stem=='mse_overview':return '主曲线：固定共同初态；每行使用当前最长可用完整窗口（不是全体初态结果）'
        if p.stem=='mse_fixed_10d':return '固定 10 天共同初态：保留真实的早期非单调变化'
        if p.stem=='mse_available_case':return '辅助曲线：可用初态随日期变化；换样本处断线，不连接虚假改善'
        if p.stem.endswith('_change_from_initial'):return '减去初始场：参考 − 初始、预测 − 初始；误差仍为预测 − 参考 · '+p.stem
        if p.stem.endswith('_el_u_v_fields'):return '原始场：初始、参考、预测及误差 · '+p.stem
        return p.stem
    def figure_markup(p):
        panel=f'<figure><img src="{p.name}" alt="{p.stem}" loading="lazy"><figcaption>{figure_label(p)} · <a href="{p.with_suffix(".svg").name}">SVG</a> · <a href="{p.with_suffix(".pdf").name}">PDF</a></figcaption></figure>'
        if '_all_methods_' in p.stem:
            return f'<details><summary>{figure_label(p)}</summary>{panel}</details>'
        return panel
    gallery=''.join(figure_markup(p) for p in figures)
    body=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>NeuralPOM · 成对冻结评估</title><style>
    :root{{--ink:#21333f;--muted:#647782;--line:#dce4e8;--accent:#34777d;--paper:#f4f6f5}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.75 system-ui,-apple-system,"PingFang SC",sans-serif}}main{{max-width:1440px;margin:auto;padding:50px 36px 90px}}header{{border-top:5px solid var(--accent);padding:28px 0 36px}}.eyebrow{{font-size:12px;letter-spacing:.15em;color:var(--accent)}}h1{{font-size:clamp(32px,4vw,58px);line-height:1.2;letter-spacing:-.04em;margin:14px 0 20px}}h2{{font-size:25px;margin:45px 0 16px}}h3{{font-size:18px}}p{{max-width:1000px}}.sub,.muted{{color:var(--muted)}}.cards{{display:grid;grid-template-columns:repeat(5,1fr);gap:14px}}.card{{background:white;padding:24px;border:1px solid var(--line);border-radius:8px}}.card strong{{display:block;font-size:38px;font-weight:600}}.card span{{font-size:13px;color:var(--muted)}}.note{{background:#e5efed;border-left:4px solid var(--accent);padding:20px 24px;border-radius:0 8px 8px 0}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}.panel{{background:white;padding:24px;border:1px solid var(--line);border-radius:8px}}.scroll{{overflow:auto;background:white;border:1px solid var(--line)}}table{{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}}th,td{{padding:10px 12px;text-align:right;border-bottom:1px solid var(--line)}}th{{background:#eaf0f1}}td:nth-child(-n+4),th:nth-child(-n+4){{text-align:left}}figure{{margin:22px 0;background:white;border:1px solid var(--line);padding:18px;border-radius:8px}}img{{width:100%;height:auto;display:block}}figcaption{{font-size:12px;color:var(--muted);margin-top:14px}}a{{color:var(--accent)}}footer{{border-top:1px solid var(--line);margin-top:50px;padding-top:24px;font-size:13px}}@media(max-width:800px){{main{{padding:24px 16px}}.cards{{grid-template-columns:repeat(2,1fr)}}.grid{{grid-template-columns:1fr}}}}
    </style></head><body><main><header><div class="eyebrow">NEURALPOM / FROZEN EVALUATION / 2026.09</div><h1>校正约束，如何改变长期预测？</h1><p class="sub">12 组冻结权重 · Raw / Constrained 成对对照 · Implicit / Explicit · 1 h / 1 day</p><p class="muted">结果快照：{now}。评估仍有 {pending} 条轨迹未结束；系统错误 {infrastructure} 条。此页面为静态快照。</p></header>
    <section class="cards">{cards}</section><h2>01 / 当前证据与初步判断</h2><div class="note"><b>分析：</b>{conclusion}<br>短期 MSE、数值稳定、长期动能统计是三个不同问题。待完整窗口覆盖后再比较长期表现。</div>
    <div class="grid"><section class="panel"><h3>已核验事实</h3><ul><li>训练已停止。日频 Post-train：Explicit 第 4846 步，Implicit 第 5038 步；小时频两者均为第 1780 步。</li><li>16 个起点：成员 7，模型时间 8000–8480，间隔 32 天。包含原来的 8 个起点。</li><li>同一权重分别运行裸校正与历史约束接口，另有 4 条配套纯物理配置。</li><li>全部推进均关闭 Triton、编译核和推理快路径；两种校正策略采用相同计算方式。</li></ul></section><section class="panel"><h3>解释边界</h3><ul><li>Raw 移除了训练时存在的投影/限制，属于推理分布变化。</li><li>Constrained 保留速度零柱平均、温盐逐层收支投影及 5 倍尺度上限，没有额外谱滤波。</li><li>不同核心使用各自生成的真值，不能将绝对误差差异直接解释为核心的因果优劣。</li><li>重叠起点不是独立训练重复；部分日期已有开发评估历史。</li></ul></section></div>
    <h2>02 / 预览：前 10 天 MSE</h2><p>只有模型与物理基线均覆盖完整 10 天的同初态轨迹才进入下表。MSE 使用物理单位平方；缺数据记为“—”。完成数指完整 300 天。失败数分母为已启动初态数；存活样本的条件均值不能代表全部初态。</p><div class="scroll"><table><thead><tr>{''.join('<th>'+x+'</th>' for x in ['核心','频率','阶段','策略','完成/16','数值失败/已启动','配对 n','U MSE','V MSE','T MSE','S MSE','SSH MSE'])}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
    <h2>03 / 曲线、能谱与空间结构</h2>{audit_note}<p>MSE 为主要指标。主曲线固定同一核心/频率内所有配置共同完整覆盖的初态集合，在所有日期和所有阶段/策略之间保持不变，图中注明窗口及 n/16；另列固定 10 天曲线。样本较少的长窗口仅为条件均值，不代表全部 16 初态。辅助 available-case 图保留全部可用记录，在初态集合变化处断线并标记竖线，跨段不可解释为误差改善。每个日期的初态及各固定窗口的初态清单见 summary.json 的 daily.cases / curve_windows；图中 raw 用虚线，constrained 用实线。空间场仅展示 EL、U、V，固定取首个起点，U/V 为第 0 层。每组依次展示原始场和减去初始场两个版本，使用相同起点与预测日期。变化场逐点计算参考 − 初始、预测 − 初始；两者之差仍等于预测 − 参考。全部使用 RdBu_r；每个版本内，同变量的初始/参考/预测场或其变化量共用以零为中心的对称色标（绝对值第 99 百分位），误差场使用独立对称色标。色标裁切仅影响显示，不改变 MSE。能谱要求所显示配置覆盖同一完整时间窗。</p>{gallery}
    <h2>04 / 数值异常与运行状态</h2><ul>{''.join(failures) or '<li>当前记录中没有数值异常或系统错误。尚未启动及运行中的轨迹不计为成功。</li>'}</ul>
    <h2>05 / 来源与复现</h2><p><a href="summary.json">分窗口汇总 JSON（50 / 200 / 300 天）</a> · <a href="daily_mse.csv">逐初态逐日 MSE CSV</a> · <a href="protocol.json">评估协议</a> · <a href="registry.json">权重、训练配置与执行策略登记</a> · <a href="evaluation_explorer.ipynb">交互式 Notebook</a></p><p>空间 MSE 按参考体积 / 面积加权；日与初态等权平均。能谱使用原生交错速度网格、湿点掩码与 Hann 窗，先平均能量再作比较。首次异常为观测到的物理推进/校正异常位置，系统错误单独标记。</p><footer>源数据优先 · 覆盖度透明 · 假设由结果检验<br>报告不会自动启动、停止或恢复计算。</footer></main></body></html>'''
    body=body.replace('<section class="cards">'+cards+'</section>','<section class="cards">'+cards+'</section>'+progress_tables(summary,protocol))
    body=body.replace('<h2>03 /',long_window_tables(summary,protocol)+pair_table+'<h2>03 /')
    body=body.replace('<div class="grid">','<p class="note">'+evidence_notes(summary)+'</p><div class="grid">',1)
    if 'density75' in protocol['correction_policies']:
        body=body.replace('<div class="grid">','<section class="panel">'+optimized_notes(summary)+'</section><div class="grid">',1)
        optimized_rows=[];raw_optimized_rows=[]
        for name,periods in summary['optimized_pairs'].items():
            p=periods['10']['optimized_vs_standard']
            q=periods['10']['raw_vs_optimized']
            values=[name,p['paired_cases'],q['paired_cases']]
            for f in ('u','v','t','s','el'):
                x=p['metrics'][f]['relative_mse_change']
                values.append('—' if x is None else f'{100*x:+.1f}%')
            optimized_rows.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in values)+'</tr>')
            raw_values=[name,q['paired_cases']]
            for f in ('u','v','t','s','el'):
                x=q['metrics'][f]['relative_mse_change']
                raw_values.append('—' if x is None else f'{100*x:+.1f}%')
            raw_optimized_rows.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in raw_values)+'</tr>')
        table='<h3>保留优化方案 density75：前 10 天相对基础约束的 MSE 变化</h3><p>负值表示优化方案误差更低。density75 在训练约束之后，将按 EOS 梯度分解的密度方向增量保留 75%，同时满足逐层温盐收支约束并再次联合限幅；不使用空间滤波。历史改善证据来自小时 Implicit Post-train 的 14 个起点。对 Explicit、日频和其他阶段的应用是本次固定参数迁移检验。</p><div class="scroll"><table><thead><tr>'+''.join('<th>'+x+'</th>' for x in ['权重','优化/标准 n','裸/优化 n','U','V','T','S','SSH'])+'</tr></thead><tbody>'+''.join(optimized_rows)+'</tbody></table></div>'
        raw_table='<h3>主对照：裸推理相对 density75 的前 10 天 MSE 变化</h3><p>正值表示裸推理误差更大；只比较同权重、同初态、相同完整时间窗。尚无完整窗口的条目保留为空，不用基础约束替代优化版。</p><div class="scroll"><table><thead><tr>'+''.join('<th>'+x+'</th>' for x in ['权重','共同 n','U','V','T','S','SSH'])+'</tr></thead><tbody>'+''.join(raw_optimized_rows)+'</tbody></table></div>'
        body=body.replace('<h2>03 /',raw_table+table+'<h2>03 /')
        body=body.replace('12 组冻结权重 · Raw / Constrained 成对对照','12 组冻结权重 · Raw / 基础约束 / Density75')
        body=body.replace('图中 raw 用虚线，constrained 用实线','物理基线 B0–B3 统一用虚线，混合模型统一用实线；Pre/Mid/Post 分别为紫/蓝/橙色系，同阶段 raw、基础约束、density75 依次由浅至深，颜色含义在 MSE 与能谱图中一致。当前尚未完成正式测试的新增基线不伪造曲线')
        body=body.replace('同一权重分别运行裸校正与历史约束接口，另有 4 条配套纯物理配置。','同一权重运行裸校正、基础约束与保留优化方案 density75，另有 4 条配套纯物理配置。')
        body=body.replace('两种校正策略采用相同计算方式','三种校正策略采用相同物理计算方式')
        body=body.replace('两分支都关闭计算加速','三分支都关闭物理计算加速')
    terminal=pending==0 and infrastructure==0 and not summary['worker_errors']
    phase='全部轨迹已到达终态，最终分析待核验' if terminal else '持续评估中 · 阶段性结果'
    body=body.replace('<h1>校正约束，如何改变长期预测？</h1>', '<h1>校正约束，如何改变长期预测？</h1><p><b>'+phase+'</b></p>')
    body=body.replace('此页面为静态快照。','此页面为带时间戳的快照；后续仅在用户主动要求时更新。')
    body=body.replace('</style>', 'details{margin:16px 0;padding:18px;background:white;border:1px solid var(--line);border-radius:8px}summary{cursor:pointer}h4{margin:24px 0 12px}</style>')
    (destination/'index.html').write_text(body)
    atomic_json(destination/'protocol.json',protocol)
    registry={}
    for name in summary['groups']:
        p=Path(protocol['output_root'])/'runs'/name/'manifest.json'
        if p.exists():registry[name]=read_json(p)
    atomic_json(destination/'registry.json',registry)
    from .protocol import sha256
    atomic_json(destination/'build.json',dict(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha256(protocol_path),
        analysis_sources={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
        artifacts={p.name:sha256(p) for p in destination.iterdir()
                   if p.is_file() and p.suffix in ('.html','.json','.csv') and p.name!='build.json'},
        snapshot_note='Summary JSON and daily CSV use one captured set of committed records. Figures may read later committed states; field leads and spectrum windows are labelled.'))
    return destination/'index.html'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();print(render(args.protocol,args.out))

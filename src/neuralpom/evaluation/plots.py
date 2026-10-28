"""Python figures with common visual semantics and explicit sample coverage."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .protocol import FIELDS, case_root
from .analysis import records

COLORS = {'physics':'#77818b','pretrain':'#a895bf','midtrain4':'#6b9dad','posttrain':'#cd937b'}
LINE_STYLES = {'raw':'-','constrained':'-','density75':'-','physics':'--'}
BASELINE_COLORS = {'B0':'#77818b', 'B1':'#b58900', 'B2':'#19836c', 'B3':'#334fa3'}
HYBRID_COLORS = {
    'pretrain': {'raw':'#c4a7d6', 'constrained':'#9464ad', 'density75':'#653481'},
    'midtrain4': {'raw':'#7ebfd0', 'constrained':'#3b92aa', 'density75':'#165c76'},
    'posttrain': {'raw':'#efa889', 'constrained':'#cf7049', 'density75':'#9d3c23'},
}
LABELS = {'u':'U (m² s⁻²)', 'v':'V (m² s⁻²)', 't':'Temperature (°C²)', 's':'Salinity (psu²)', 'el':'SSH (m²)'}


def curve_appearance(variant):
    if variant.get("model_family") == "ai":
        return (f"AI · {variant['stage']}", "#176b53" if variant["stage"] == "posttrain" else "#70aa98", ":", 1.7)
    if variant['stage'] == 'physics':
        family = variant.get('baseline_family', 'B0')
        return family + ' · physics', BASELINE_COLORS[family], '--', 1.5
    return (f"{variant['stage']} · {variant['policy']}", HYBRID_COLORS[variant['stage']][variant['policy']],
            LINE_STYLES[variant['policy']], 1.4 if variant['stage']=='posttrain' else 1.05)


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
        'axes.labelsize':9,'axes.spines.top':False,'axes.spines.right':False,
        'axes.edgecolor':'#87909b','axes.linewidth':.6,'grid.color':'#e7eaef',
        'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42,
        'figure.facecolor':'white','axes.facecolor':'white'})


def save(fig, path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','svg','pdf'):
        fig.savefig(path.with_suffix('.'+suffix),dpi=180,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    return path.with_suffix('.png')


def support_segments(days):
    """Never draw a line across a change in the actual origin set."""
    segments = []
    last = None
    for day in days:
        support = tuple((c['member'], c['start_day']) for c in day['cases'])
        if support != last:
            segments.append([])
        segments[-1].append(day)
        last = support
    return segments


def plot_mse(summary, destination, mode='fixed', horizon=None):
    if mode not in ('fixed','available','registered'):
        raise ValueError('Unknown MSE curve mode')
    style()
    groups=list(dict.fromkeys(g['variant']['group_id'] for g in summary['groups'].values()))
    fig,axes=plt.subplots(len(groups),5,figsize=(17,2.8*len(groups)),squeeze=False,layout='constrained')
    for row,group in enumerate(groups):
        windows=summary['registered_curve_windows' if mode=='registered' else 'curve_windows'][group]
        candidates=[int(h) for h,w in windows.items() if w['n'] and (horizon is None or int(h)==horizon)]
        window=windows[str(max(candidates))] if candidates else None
        for col,field in enumerate(FIELDS):
            ax=axes[row,col]
            for name,item in summary['groups'].items():
                v=item['variant']
                if v['group_id']!=group:continue
                days=item['daily'] if mode=='available' else window['daily'][name] if window else []
                label,color,line_style,width=curve_appearance(v)
                if not days:
                    if mode=='registered':
                        reference=window or next(iter(windows.values()))
                        reason=reference['unavailable_variants'].get(name,'no complete common window')
                        ax.plot([],[],color=color,ls=line_style,lw=width,label=label+' ('+reason+')')
                    continue
                segments=support_segments(days) if mode=='available' else [days]
                for i,segment in enumerate(segments):
                    ax.plot([d['lead_day'] for d in segment],[d['mse'][field] for d in segment],
                        color=color,ls=line_style,
                        lw=width,label=label if i==0 else None,
                        marker='o' if len(segment)==1 else None,markersize=3)
                    if mode=='available' and i:
                        d=segment[0]
                        ax.plot(d['lead_day'],d['mse'][field],marker='|',color=color,ms=6)
            ax.set_title(LABELS[field]);ax.grid(axis='y',alpha=.7)
            ax.set_xlabel('Lead day');ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,3))
            if col==0:
                support='Variable origins; breaks at changes' if mode=='available' else (
                    f"Fixed n={window['n']}/16; days 1–{window['horizon']}" if window else 'No complete common window')
                if mode=='registered':
                    reference=window or next(iter(windows.values()))
                    support+=f"\n{len(reference['included_variants'])}/{len(reference['requested_variants'])} registered methods"
                ax.set_ylabel(group.replace('implicit_p0_fixed','Implicit').replace('explicit','Explicit')+'\n'+support+'\nMSE')
            if col==4 and ax.lines:ax.legend(fontsize=6,loc='upper left',bbox_to_anchor=(1.02,1))
    kind=('Available-case diagnostic | lines break when origins change; segments are not directly comparable'
          if mode=='available' else 'Fixed common origins across ALL variants and lead days within each row | partial coverage, conditional means')
    if mode=='registered':
        kind='Provisional registered-method cohort | missing methods labelled; registered failures remain in the full-window gate'
    fig.suptitle('Daily physical-unit MSE | dashed: physics baselines · solid: hybrid models · dotted: pure AI\n'+kind,fontsize=12)
    filename='mse_available_case' if mode=='available' else f'mse_registered_{horizon}d' if mode=='registered' else f'mse_fixed_{horizon}d' if horizon else 'mse_overview'
    return save(fig,Path(destination)/filename)


def plot_coverage(summary,destination):
    style();items=list(summary['groups'].items())
    fig,axes=plt.subplots(1,2,figsize=(14,10),layout='constrained',sharey=True)
    labels=[]
    for i,(name,item) in enumerate(items):
        v=item['variant'];labels.append(name.replace('implicit_p0_fixed','implicit'))
        axes[0].barh(i,item['completed_case_days'],color=COLORS[v['stage']])
        s=item['statuses'];left=0
        for key,color in [('complete','#6b9dad'),('running','#bfcdd5'),('numerical_failure','#c77870'),('infrastructure_error','#a895bf'),('pending','#eef0f3')]:
            axes[1].barh(i,s[key],left=left,color=color,label=key if i==0 else None);left+=s[key]
    axes[0].set_yticks(range(len(labels)),labels,fontsize=7);axes[0].invert_yaxis()
    axes[0].set_xlabel('Committed case-days');axes[1].set_xlabel('Initial conditions')
    axes[1].legend(loc='upper left',bbox_to_anchor=(1.01,1),fontsize=8)
    axes[0].set_title('Exposure');axes[1].set_title('Outcome at requested horizon')
    fig.suptitle('Coverage and failures must accompany MSE',fontsize=15)
    return save(fig,Path(destination)/'coverage')


def result_case_root(protocol,variant_id,case,run_roots=None):
    """Resolve separately registered runs after caller-side provenance checks.

Overrides point to a variant's run directory, not a calibration campaign or
its case directory. This helper does not certify reference/frequency reuse.
"""
    if run_roots is not None and variant_id in run_roots:
        from .protocol import case_id
        return Path(run_roots[variant_id])/case_id(case)
    return case_root(protocol['output_root'],variant_id,case)


def plot_fields(protocol,variant_id,case,day,field='t',level=0,subtract_initial=False,
                cmap='viridis',limits=None,symmetric_clim=False,run_roots=None):
    """One field or a field sequence; shared state limits within each variable."""
    style();root=result_case_root(protocol,variant_id,case,run_roots)
    fields=(field,) if isinstance(field,str) else tuple(field)
    units={'el':'EL / SSH (m)','u':'U (m s⁻¹)','v':'V (m s⁻¹)',
           't':'Temperature (°C)','s':'Salinity (psu)'}
    fig,axes=plt.subplots(len(fields),4,figsize=(13,3.5*len(fields)),squeeze=False,layout='constrained')
    for row,name in enumerate(fields):
        def load(filename):
            with np.load(root/filename) as values:a=values[name].copy()
            return a if a.ndim==2 else a[...,level]
        initial=load('initial.npz');ref=load(f'day{day:03d}_reference.npz');pred=load(f'day{day:03d}_post.npz')
        with np.load(root/'geometry.npz') as geo:
            mask=geo['dum' if name=='u' else 'dvm' if name=='v' else 'fsm']>.5
        arrays=[initial,ref,pred]
        if subtract_initial:arrays=[a-initial for a in arrays]
        arrays=[np.where(mask,a,np.nan) for a in arrays]
        finite=np.concatenate([a[np.isfinite(a)] for a in arrays])
        if symmetric_clim:
            bound=max(float(max(abs(x) for x in limits) if limits else np.quantile(np.abs(finite),.99)),1e-12)
            lo,hi=-bound,bound
        else:
            lo,hi=limits or np.quantile(finite,[.01,.99])
            if hi<=lo:hi=lo+1e-8
        error=np.where(mask,pred-ref,np.nan);bound=max(float(np.nanquantile(np.abs(error),.99)),1e-12)
        titles=('Initial − initial (0)','Reference − initial','Prediction − initial') if subtract_initial else ('Initial','Reference','Prediction')
        for ax,a,title in zip(axes[row,:3],arrays,titles):
            im=ax.imshow(a.T,origin='lower',cmap=cmap,vmin=lo,vmax=hi,aspect='auto')
            ax.set_title(f'{units[name]} | {title}')
        fig.colorbar(im,ax=list(axes[row,:3]),shrink=.75,label=units[name])
        im=axes[row,3].imshow(error.T,origin='lower',cmap='RdBu_r',vmin=-bound,vmax=bound,aspect='auto')
        axes[row,3].set_title(f'{units[name]} | Prediction − reference')
        fig.colorbar(im,ax=axes[row,3],shrink=.75,label=units[name])
        for ax in axes[row]:ax.set_xlabel('Grid i');ax.set_ylabel('Grid j')
    quantity='change' if subtract_initial else 'state'
    mode='Change from initial' if subtract_initial else 'Absolute fields'
    limit_note=(f'Shared symmetric ±99% |{quantity}| limits per variable' if symmetric_clim else 'Shared 1–99% state limits')
    fig.suptitle(f"{variant_id} | member {case['member']}, start {case['start_day']}, lead {day} d | {mode} | {', '.join(fields)}; 3D level {level}\n{limit_note}; separate symmetric 99% error limits (display only)",fontsize=10)
    return fig


def plot_spectra(protocol,variant_ids,case,first,last,representation='depth_weighted',run_roots=None):
    style();fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained');reference_drawn=False
    for name in variant_ids:
        root=result_case_root(protocol,name,case,run_roots)
        path=root/'status.json'
        if not path.exists():continue
        from .protocol import read_json
        status=read_json(path)
        rows=[r for r in records(root/'spectra.jsonl',status['completed_days']) if first<=r['lead_day']<=last]
        if len(rows)!=last-first+1:continue
        def spectrum(key):
            a=np.mean([r[key] for r in rows],axis=0)
            if representation=='depth_weighted':return (a[:15]*np.asarray(rows[0]['sigma_weights'])[:,None]).sum(0)
            return a[15 if representation=='depthmean' else int(representation)]
        k=np.asarray(rows[0]['wavenumber'])
        if not reference_drawn:
            ax.loglog(k,spectrum('reference'),color='#283746',lw=2,label='Reference');reference_drawn=True
        stage=next(s for s in COLORS if '_'+s+'_' in name)
        policy=name.split('_')[-1]
        variant=dict(stage=stage,policy='physics' if stage=='physics' else policy)
        if stage=='physics':variant['baseline_family']=policy if policy in BASELINE_COLORS else 'B0'
        label,color,line_style,width=curve_appearance(variant)
        ax.loglog(k,spectrum('prediction'),color=color,ls=line_style,lw=width,label=label)
    ax.set_xlabel('Wavenumber (cycles per grid interval)');ax.set_ylabel('Shell-summed kinetic energy (m² s⁻²)')
    ax.set_title(f"Native-grid KE | days {first}–{last} | {representation}\nHann window; wet-mask mean removed; identical reference window")
    ax.legend(fontsize=8);ax.grid(alpha=.4,which='both')
    return fig


def plot_field_comparison(protocol,variant_ids,case,day,field='el',level=0,
                          subtract_initial=False,run_roots=None):
    """All requested methods in one spatial figure, with explicit missing slots.

The caller verifies scientific provenance. Here we additionally require exact
initial/reference fields and wet masks across every available method.
"""
    if field not in ('el','u','v') or not variant_ids or len(set(variant_ids))!=len(variant_ids):
        raise ValueError('Unique methods and EL/U/V are required')
    groups={name.split('_'+next(s for s in COLORS if '_'+s+'_' in name)+'_',1)[0] for name in variant_ids}
    if len(groups)!=1:raise ValueError('One core/frequency group per spatial comparison is required')
    from .protocol import read_json
    style();panels=[];shared=None
    for name in variant_ids:
        root=result_case_root(protocol,name,case,run_roots)
        status=read_json(root/'status.json') if (root/'status.json').exists() else dict(status='pending',completed_days=0)
        stage=next(s for s in COLORS if '_'+s+'_' in name)
        policy=name.split('_')[-1]
        variant=dict(stage=stage,policy='physics' if stage=='physics' else policy)
        if stage=='physics':variant['baseline_family']=policy if policy in BASELINE_COLORS else 'B0'
        label,color,_,_=curve_appearance(variant)
        if status['completed_days']<day:
            panels.append((label,None,color,f"Unavailable: {status.get('status','unknown')}\n{status['completed_days']} committed days"))
            continue
        def load(filename):
            with np.load(root/filename) as values:a=values[field].copy()
            return a if a.ndim==2 else a[...,level]
        initial,reference,prediction=load('initial.npz'),load(f'day{day:03d}_reference.npz'),load(f'day{day:03d}_post.npz')
        with np.load(root/'geometry.npz') as geometry:
            mask=geometry['dum' if field=='u' else 'dvm' if field=='v' else 'fsm']>.5
        if any(a.shape!=mask.shape or not np.isfinite(a[mask]).all() for a in (initial,reference,prediction)):
            raise ValueError('Nonfinite or mismatched committed spatial field')
        if shared is None:shared=(initial,reference,mask)
        elif not all(np.array_equal(a,b,equal_nan=True) for a,b in zip(shared,(initial,reference,mask))):
            raise ValueError('Spatial comparison has different initial/reference fields or masks')
        values=prediction-initial if subtract_initial else prediction
        panels.append((label,np.where(mask,values,np.nan),color,None))
    if shared is None:raise ValueError('No reference field available at requested day')
    initial,reference,mask=shared
    prefix=[('Initial',initial),('Reference',reference)]
    prefix=[(label+(' − initial' if subtract_initial else ''),
             np.where(mask,a-initial if subtract_initial else a,np.nan),'#283746',None)
            for label,a in prefix]
    panels=prefix+panels
    finite=np.concatenate([a[np.isfinite(a)] for _,a,_,_ in panels if a is not None])
    bound=max(float(np.quantile(np.abs(finite),.99)),1e-12)
    columns=4;rows=(len(panels)+columns-1)//columns
    fig,axes=plt.subplots(rows,columns,figsize=(14,3.2*rows),squeeze=False,layout='constrained')
    used=[];image=None
    for ax,(label,a,color,note) in zip(axes.flat,panels):
        ax.set_title(label,color=color,fontsize=9)
        if a is None:
            ax.text(.5,.5,note,ha='center',va='center',transform=ax.transAxes);ax.set_axis_off()
        else:
            image=ax.imshow(a.T,origin='lower',cmap='RdBu_r',vmin=-bound,vmax=bound,aspect='auto')
            ax.set_xlabel('Grid i');ax.set_ylabel('Grid j');used.append(ax)
    for ax in list(axes.flat)[len(panels):]:ax.set_axis_off()
    units='m' if field=='el' else 'm s⁻¹'
    fig.colorbar(image,ax=used,shrink=.8,label=f'{field.upper()} ({units})')
    mode='Change from same initial' if subtract_initial else 'Absolute fields'
    fig.suptitle(f"{next(iter(groups))} | {field.upper()} | member {case['member']}, start {case['start_day']}, lead {day} d | {mode}\nAll methods: shared symmetric ±99% limits (display only); level {level}",fontsize=11)
    return fig

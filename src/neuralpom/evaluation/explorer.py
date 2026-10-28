"""Read-only Jupyter controls. Importing this module never runs inference."""
from pathlib import Path

from .protocol import load_protocol, variants, read_json, case_root
from .analysis import collect, summarize


def explorer(protocol_path, physics_base=None):
    import ipywidgets as w
    from IPython.display import display, clear_output
    import matplotlib.pyplot as plt
    from .plots import plot_fields, plot_spectra, plot_field_comparison, curve_appearance, support_segments, result_case_root
    protocol=load_protocol(protocol_path)
    physics_base=Path(physics_base) if physics_base is not None else Path(protocol['output_root']).parent/'physics_baselines_20260927'
    def snapshot():
        if physics_base.exists():
            from .formal_registry import collect_with_formal
            return collect_with_formal(protocol,physics_base)
        return collect(protocol),{},{}
    initial_data,_,_=snapshot()
    choices=[item['variant'] for item in initial_data.values()]
    model=w.Dropdown(options=[v['id'] for v in choices],description='Model',layout=w.Layout(width='600px'))
    origin=w.Dropdown(options=[c['start_day'] for c in protocol['cases']],description='Start day')
    lead=w.IntSlider(min=1,max=max(protocol['horizons_days']),value=1,description='Lead day',continuous_update=False)
    field=w.Dropdown(options=['u','v','t','s','el'],value='el',description='Variable')
    level=w.IntSlider(min=0,max=14,value=0,description='Level',continuous_update=False)
    anomaly=w.Checkbox(value=False,description='Subtract initial')
    cmap=w.Dropdown(options=['RdBu_r','viridis','cividis','plasma'],description='Colormap')
    low=w.Text(value='',placeholder='auto 1%',description='Min')
    high=w.Text(value='',placeholder='auto 99%',description='Max')
    view=w.ToggleButtons(options=['Fields','All fields','KE spectra','MSE'],description='View')
    first=w.IntText(value=1,description='Spectrum from')
    last=w.IntText(value=10,description='to day')
    representation=w.Dropdown(options=['depth_weighted','depthmean',*range(15)],description='KE level')
    button=w.Button(description='Refresh / draw',button_style='info',icon='refresh')
    output=w.Output()
    def draw(_=None):
        with output:
            clear_output(wait=True)
            try:
                records,run_roots,provenance=snapshot()
                case=dict(member=7,start_day=origin.value)
                root=result_case_root(protocol,model.value,case,run_roots)
                status=read_json(root/'status.json') if (root/'status.json').exists() else {'status':'pending','completed_days':0}
                print(f"{status['status']} | {status['completed_days']} committed days | member 7 / seed 0")
                if provenance:
                    print('Formal tests only; unavailable/daily reuse slots remain pending. CPU/CUDA grid rounding differs; aliases are not independent samples.')
                if view.value=='Fields':
                    if lead.value>status['completed_days']:
                        print('This lead is not available. Select a committed lead or refresh later.');return
                    limits=(float(low.value),float(high.value)) if low.value and high.value else None
                    if limits and limits[0]>=limits[1]:raise ValueError('Min must be smaller than Max')
                    fig=plot_fields(protocol,model.value,case,lead.value,field.value,level.value,anomaly.value,cmap.value,limits,
                        symmetric_clim=cmap.value=='RdBu_r',run_roots=run_roots)
                elif view.value=='All fields':
                    if field.value not in ('el','u','v'):
                        raise ValueError('All-method spatial comparison supports EL, U and V')
                    selected=next(v for v in choices if v['id']==model.value)
                    names=[v['id'] for v in choices if v['group_id']==selected['group_id']]
                    fig=plot_field_comparison(protocol,names,case,lead.value,field.value,level.value,anomaly.value,run_roots=run_roots)
                    print('Shared symmetric RdBu_r limits; unavailable methods remain labelled. Single-origin example.')
                elif view.value=='KE spectra':
                    selected=next(v for v in choices if v['id']==model.value)
                    names=[v['id'] for v in choices if v['group_id']==selected['group_id']]
                    if first.value<1 or last.value<first.value:raise ValueError('Invalid spectral time window')
                    fig=plot_spectra(protocol,names,case,first.value,last.value,representation.value,run_roots=run_roots)
                else:
                    data=summarize(protocol,records);selected=next(v for v in choices if v['id']==model.value)
                    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
                    for item in data['groups'].values():
                        v=item['variant']
                        if v['group_id']!=selected['group_id']:continue
                        rows=item['daily'];label,color,line_style,width=curve_appearance(v)
                        if not rows:
                            axes[1].plot([],[],label=label+' (unavailable)',color=color,ls=line_style,lw=width)
                            continue
                        for index,segment in enumerate(support_segments(rows)):
                            axes[0].plot([r['lead_day'] for r in segment],[r['mse'][field.value] for r in segment],
                                label=label if index==0 else None,color=color,ls=line_style,lw=width,
                                marker='o' if len(segment)==1 else None,markersize=3)
                        axes[1].plot([r['lead_day'] for r in rows],[r['n'] for r in rows],label=label,
                            color=color,ls=line_style,lw=width)
                    axes[0].set_ylabel(field.value+' MSE (physical units squared)');axes[1].set_ylabel('Available initial conditions')
                    for ax in axes:ax.set_xlabel('Lead day');ax.grid(alpha=.2)
                    axes[1].legend(fontsize=7);fig.suptitle('Available-case diagnostic | breaks at origin-set changes\nDashed: physics; solid: hybrids. Segments are not directly comparable.')
                display(fig);plt.close(fig)
            except (FileNotFoundError,ValueError) as error:
                print(type(error).__name__+': '+str(error))
    button.on_click(draw)
    controls=w.VBox([model,w.HBox([view,button]),w.HBox([origin,lead,field]),
        w.HBox([level,anomaly,cmap]),w.HBox([low,high]),w.HBox([first,last,representation])])
    display(controls,output)
    draw()
    return controls

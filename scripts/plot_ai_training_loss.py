#!/usr/bin/env python3
"""Snapshot current complete local loss records and sparse online validation."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def read_complete(path):
    raw=Path(path).read_bytes()
    lines=raw.decode().splitlines();rows=[]
    for index,line in enumerate(lines):
        if not line.strip():continue
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index!=len(lines)-1:raise
    if [r["step"] for r in rows]!=list(range(1,len(rows)+1)):
        raise ValueError("Loss records are missing, reordered or duplicated")
    if not rows or not all(np.isfinite(r['loss']) and r['loss']>0 for r in rows):
        raise ValueError("Loss snapshot must contain positive, finite measurements")
    return rows,hashlib.sha256(raw).hexdigest()


def moving_mean(values,window):
    sums=np.concatenate(([0.],np.cumsum(values,dtype=np.float64)))
    ends=np.arange(1,len(values)+1);starts=np.maximum(0,ends-window)
    return (sums[ends]-sums[starts])/(ends-starts)


def validation_rows(root,stage,metadata):
    import wandb
    tracking=json.loads((root/stage/'tracking.json').read_text())
    entity=metadata['wandb'].get('entity')
    if not entity:raise ValueError('Set the W&B entity for unambiguous validation provenance')
    run=wandb.Api(timeout=30).run(f"{entity}/{tracking['project']}/{tracking['id']}")
    records=run.history(samples=10000,keys=['validation/loss'],pandas=False)
    result=sorted((dict(step=int(r['_step']),loss=float(r['validation/loss']))
                   for r in records if r.get('validation/loss') is not None),key=lambda r:r['step'])
    expected=json.loads((root/stage/'epoch.json').read_text()) if (root/stage/'epoch.json').exists() else None
    if expected and not any(r['step']==expected['step'] for r in result):
        # Latest cloud upload can lag a completed local epoch.
        result.append(dict(step=expected['step'],loss=expected['validation_loss'],source='local latest epoch'))
        result.sort(key=lambda r:r['step'])
    if expected and len(result)!=expected['epoch']:
        raise ValueError(f"Incomplete validation history for {stage}: {len(result)} of {expected['epoch']}")
    if any(not np.isfinite(r['loss']) or r['loss']<=0 for r in result):
        raise ValueError('Nonpositive/nonfinite validation loss')
    return result,tracking['url']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--window',type=int,default=200)
    args=parser.parse_args()
    if args.window<1:raise ValueError('Rolling window must be positive')
    root=Path(args.run_root);output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    snapshot=datetime.now(ZoneInfo('Asia/Shanghai'))
    stages=[name for name in ('pretrain','posttrain') if (root/name/'metrics.jsonl').exists()]
    data={};summary=dict(snapshot_china=snapshot.isoformat(),run_root=str(root),rolling_window=args.window,
        objective='equal normalized complete-state channels, wet reference geometry',
        smoothing='trailing mean, initial windows use available observations; no uncertainty band',stages={})
    for stage in stages:
        rows,digest=read_complete(root/stage/'metrics.jsonl')
        config=json.loads((root/stage/'config.json').read_text())
        validation,url=validation_rows(root,stage,config)
        steps=np.array([r['step'] for r in rows]);loss=np.array([r['loss'] for r in rows])
        # Real sampler budget, including Posttrain padding and ten daily updates.
        from neuralpom.ai_baselines.config import plan
        per_epoch=plan(config)[stage]['updates_per_epoch']
        smoothed=moving_mean(loss,args.window)
        median=np.array([np.median(loss[max(0,i-args.window+1):i+1]) for i in range(len(loss))])
        completed=(root/stage/'COMPLETE.json').exists()
        info=dict(points=len(rows),latest_step=int(steps[-1]),updates_per_epoch=per_epoch,
            epoch_progress=float(steps[-1]/per_epoch),completed=completed,
            recent_200_step_mean=float(smoothed[-1]),initial_200_step_mean=float(loss[:min(args.window,len(loss))].mean()),
            latest_raw_loss=float(loss[-1]),validation=validation,wandb_url=url,
            maximum_raw_loss=float(loss.max()),count_above_100=int((loss>100).sum()),
            count_above_zoom=int((loss>5).sum()),median_raw_loss=float(np.median(loss)),
            recent_200_step_median=float(median[-1]),
            loss_log_sha256=digest,validation_horizon_days=1 if stage=='pretrain' else 10)
        summary['stages'][stage]=info;data[stage]=(steps,loss,smoothed,median,validation,info)
        with (output/f'{stage}_loss.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['step','epoch','raw_loss','trailing_mean','trailing_median'])
            writer.writerows(zip(steps,steps/per_epoch,loss,smoothed,median))
        with (output/f'{stage}_validation.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['step','epoch','validation_loss'])
            writer.writerows((r['step'],r['step']/per_epoch,r['loss']) for r in validation)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.7,'svg.fonttype':'none','pdf.fonttype':42,
        'legend.frameon':False,'figure.facecolor':'white','axes.facecolor':'white'})
    fig,axes=plt.subplots(2,len(stages),figsize=(13,7.6),squeeze=False,sharex='col')
    colors={'pretrain':'#245d83','posttrain':'#157565'}
    for col,stage in enumerate(stages):
        steps,loss,smooth,median,validation,info=data[stage];color=colors[stage];x=steps/1000
        for row in (0,1):
            ax=axes[row,col]
            ax.plot(x,loss,color='#aeb8c2',alpha=.28,lw=.45,label='Raw batch loss (8-rank mean)',zorder=1,rasterized=True)
            central=smooth if row==0 else median
            label=f'Trailing {args.window}-update '+('mean' if row==0 else 'median')
            ax.plot(x,central,color=color,lw=1.65,label=label,zorder=3)
            if validation:
                vx=np.array([r['step'] for r in validation])/1000;vy=np.array([r['loss'] for r in validation])
                ax.plot(vx,vy,'o-',color='#b36e28',markersize=3.5,lw=1.1,label='Validation (epoch end)',zorder=4)
                best=int(np.argmin(vy))
                ax.scatter(vx[best],vy[best],s=72,facecolors='none',edgecolors='#b36e28',lw=1.1,zorder=5)
                if row==1:
                    best_epoch=validation[best]['step']/info['updates_per_epoch']
                    ax.text(.98,.025,f"Best validation: {vy[best]:.4f} at epoch {best_epoch:g}",
                        transform=ax.transAxes,ha='right',va='bottom',fontsize=8.5,color='#805020')
            ax.set_yscale('log');ax.grid(axis='y',which='major',color='#e2e7eb',lw=.6)
            ax.set_ylabel('Standardized full-state MSE (log scale)',fontsize=9)
            ax.set_xlim(0,float(x[-1])*1.02);ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.tick_params(axis='both',labelsize=8.5)
            ax.legend(loc='upper right',fontsize=7.8)
            if row==0:
                status='completed' if info['completed'] else 'running'
                ax.set_title(f"{stage.capitalize()} | {status} | full range",loc='left',fontweight='bold',fontsize=12,pad=35)
                text=f"{info['points']:,} updates · {info['epoch_progress']:.1f} epochs"+chr(10)+f"Validation: {info['validation_horizon_days']}-day forecast"
                ax.text(.01,.98,text,transform=ax.transAxes,va='top',fontsize=8.5,color='#435567',
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=3))
                secondary=ax.secondary_xaxis('top',functions=(lambda x,p=info['updates_per_epoch']:x*1000/p,
                    lambda x,p=info['updates_per_epoch']:x*p/1000))
                secondary.set_xlabel('Epoch',fontsize=8.5)
                secondary.xaxis.set_major_locator(MaxNLocator(5,integer=True));secondary.tick_params(labelsize=8)
            else:
                ax.set_ylim(.01,5)
                ax.set_xlabel('Optimizer updates (thousands)',fontsize=9)
                ax.set_title('Typical range | robust trend',loc='left',fontsize=10,fontweight='bold',pad=7)
                text=f"{info['count_above_zoom']} raw points above 5"+chr(10)+'retained in the full-range panel'
                ax.text(.01,.98,text,transform=ax.transAxes,va='top',fontsize=8.2,color='#435567',
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=3))
    fig.suptitle('Triton · implicit reference · seed 0 · eight GPUs',x=.07,ha='left',fontsize=15,fontweight='bold',y=.98)
    fig.text(.07,.045,'Top: full recorded range. Bottom: 0.01–5 zoom; excluded spikes remain visible above. Smoothing is descriptive.',fontsize=8.5,color='#526271')
    fig.text(.07,.022,'Pretrain and Posttrain use different validation horizons; their losses are not a paired stage comparison.',fontsize=8.5,color='#526271')
    fig.text(.98,.022,'Snapshot: '+snapshot.strftime('%Y-%m-%d %H:%M CST'),ha='right',fontsize=8,color='#526271')
    fig.subplots_adjust(left=.07,right=.98,bottom=.13,top=.84,wspace=.24,hspace=.27)
    for suffix in ('png','svg','pdf'):
        fig.savefig(output/f'triton_loss.{suffix}',dpi=180,facecolor='white')
    plt.close(fig)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({stage:{k:v[k] for k in ('latest_step','epoch_progress','completed','recent_200_step_mean','initial_200_step_mean')} for stage,v in summary['stages'].items()},indent=2))


if __name__=='__main__':main()

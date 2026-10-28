"""Five-model paired-support reports, physical diagnostics and static Python plots."""
import csv, json
from pathlib import Path
import numpy as np
from neuralpom.evaluation.protocol import atomic_json, read_json, case_root, FIELDS, sha256
from neuralpom.evaluation.analysis import records
from .config import load_config


def summarize_suite(manifest_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    manifest_path=Path(manifest_path); suite=read_json(manifest_path)
    output=Path(suite["suite_root"])/"delivery";output.mkdir(exist_ok=True)
    arms=[]
    for entry in suite["models"]:
        config=load_config(entry["config"])
        for stage in ("pretrain","posttrain"):
            root=Path(config["run_root"])/"evaluation"/(stage+"_100d")
            m=read_json(root/"manifest.json");summary=read_json(root/"summary.json")
            data={}
            for case in m["cases"]:
                directory=case_root(root,m["variant"]["id"],case)
                status=read_json(directory/"status.json")
                key=(case["member"],case["start_day"])
                data[key]=dict(status=status,daily=records(directory/"daily.jsonl",status["completed_days"]),
                    profiles=records(directory/"profiles.jsonl",status["completed_days"]),
                    spectra=records(directory/"spectra.jsonl",status["completed_days"]))
            arms.append(dict(architecture=entry["architecture"],stage=stage,root=root,manifest=m,summary=summary,data=data))
    reference=arms[0]["manifest"]["training_contract"]
    for arm in arms[1:]:
        contract=arm["manifest"]["training_contract"]
        for key in ("reference_core","data_identity","split","codec","interval_seconds","state_stats_sha256"):
            if contract[key]!=reference[key]:raise ValueError("Unequal suite evaluation contract: "+key)
        if arm["manifest"]["cases"]!=arms[0]["manifest"]["cases"]:raise ValueError("Unequal origins")
    common=set.intersection(*[{k for k,v in a["data"].items() if v["status"]["status"]=="complete" and len(v["daily"])==100} for a in arms])
    result=dict(schema="neuralpom-ai-five-model-report-v1",core=reference["reference_core"],days=100,
        requested_origins=len(arms[0]["manifest"]["cases"]),common_full_windows=len(common),common_origins=[list(k) for k in sorted(common)],
        seed_count=1,cohort="development",reserved_days_used=False,
        canonical_pretrain="validation_best",canonical_posttrain="last_completed_at_early_stop_or_epoch_budget",
        triton_posttrain_exception="latest atomically committed partial epoch after user plateau stop; historical best weights unavailable",
        training_normalization_sha256=suite["normalization_sha256"],manifest_sha256=sha256(manifest_path),
        primary_metric="same-core spatially weighted physical-unit MSE",conditional_on_common_full_windows=True,arms=[])
    metrics=[]
    for arm in arms:
        record=dict(architecture=arm["architecture"],stage=arm["stage"],evaluation_root=str(arm["root"]),
            paired_full_windows=arm["summary"]["paired_full_windows"],weight_sha256=arm["manifest"]["weight_sha256"],
            selection=arm["manifest"]["selection"],mean_mse_common={field:float(np.mean([r["metrics"]["mse/"+field] for key in common for r in arm["data"][key]["daily"]])) if common else None for field in FIELDS},
            statuses=arm["summary"]["statuses"])
        result["arms"].append(record)
        for key,value in arm["data"].items():
            for row in value["daily"]:
                metrics.append(dict(architecture=arm["architecture"],stage=arm["stage"],member=key[0],start_day=key[1],lead_day=row["lead_day"],common_full_window=key in common,**row["metrics"]))
    atomic_json(output/"FINAL_RESULTS.json",result)
    keys=sorted({k for r in metrics for k in r})
    with (output/"all_daily_metrics.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,keys);writer.writeheader();writer.writerows(metrics)
    # Every lead uses the same complete origins across all ten arms; failures remain in the report.
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    colors=dict(zip(("triton","resunet","simvp","convlstm","pastnet"),plt.cm.tab10.colors))
    for ax,field in zip(axes.flat,FIELDS):
        for arm in arms:
            if not common:continue
            curve=np.mean([[r["metrics"]["mse/"+field] for r in arm["data"][key]["daily"]] for key in sorted(common)],axis=0)
            ax.plot(np.arange(1,101),curve,label=arm["architecture"]+" "+arm["stage"],color=colors[arm["architecture"]],ls="--" if arm["stage"]=="pretrain" else "-")
        ax.set(xlabel="Forecast lead (days)",ylabel="Physical-unit MSE",title=field,yscale="log");ax.grid(alpha=.2)
    axes.flat[-1].axis("off")
    handles,labels=axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles,labels,loc="center",fontsize=9)
    fig.suptitle("AI baselines: common complete 100-day origins = "+str(len(common)))
    for extension in ("png","pdf","svg"):fig.savefig(output/("daily_mse."+extension),dpi=180)
    plt.close(fig)
    # Profile and additive spectral means are computed before any energy ratios.
    aggregates=[]
    for arm in arms:
        for lead in (1,10,30,60,100):
            if not common:continue
            pp=[arm["data"][k]["profiles"][lead-1] for k in sorted(common)]
            ss=[arm["data"][k]["spectra"][lead-1] for k in sorted(common)]
            aggregates.append(dict(architecture=arm["architecture"],stage=arm["stage"],lead_day=lead,
                profiles={key:np.mean([p[key] for p in pp],axis=0).tolist() for key in pp[0] if key!="lead_day"},
                spectra={key:np.mean([s[key] for s in ss],axis=0).tolist() if key in ("prediction","reference","error","cross") else ss[0][key] for key in ss[0] if key!="lead_day"}))
    atomic_json(output/"profiles_spectra_common_origins.json",dict(common_origins=result["common_origins"],records=aggregates))
    profile_rows, spectral_rows = [], []
    for entry in aggregates:
        identity={key:entry[key] for key in ("architecture","stage","lead_day")}
        profile=entry["profiles"];spectrum=entry["spectra"]
        for level,sigma in enumerate(profile["sigma"]):
            profile_rows.append(dict(**identity,level=level,sigma=sigma,**{key:values[level] for key,values in profile.items() if key!="sigma"}))
        for rep_index,rep in enumerate(spectrum["representations"]):
            for index,k in enumerate(spectrum["wavenumber"]):
                spectral_rows.append(dict(**identity,representation=rep,wavenumber=k,
                    **{name:spectrum[name][rep_index][index] for name in ("prediction","reference","error","cross")}))
    for name,rows in (("profiles_common_origins",profile_rows),("spectra_common_origins",spectral_rows)):
        if rows:
            with (output/(name+".csv")).open("w",newline="") as stream:
                writer=csv.DictWriter(stream,list(rows[0]));writer.writeheader();writer.writerows(rows)
    selected=[a for a in aggregates if a["lead_day"]==100 and a["stage"]=="posttrain"]
    if selected:
        fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
        for ax,field in zip(axes,("t","s","rho")):
            for entry in selected:
                pp=entry["profiles"]
                ax.plot(pp[field+"/prediction"],pp["sigma"],label=entry["architecture"],color=colors[entry["architecture"]])
            pp=selected[0]["profiles"];ax.plot(pp[field+"/reference"],pp["sigma"],color="black",ls="--",label="Reference")
            ax.set(xlabel=field,ylabel="Sigma coordinate",title="Day 100 "+field);ax.grid(alpha=.2)
        axes[-1].legend(fontsize=8)
        for extension in ("png","pdf","svg"):fig.savefig(output/("day100_profiles."+extension),dpi=180)
        plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
        for entry in selected:
            ss=entry["spectra"];index=ss["representations"].index("depthmean")
            for ax,quantity in zip(axes,("prediction","error")):
                ax.loglog(ss["wavenumber"],ss[quantity][index],label=entry["architecture"],color=colors[entry["architecture"]])
        ss=selected[0]["spectra"];index=ss["representations"].index("depthmean")
        axes[0].loglog(ss["wavenumber"],ss["reference"][index],color="black",ls="--",label="Reference")
        for ax,title in zip(axes,("Predicted energy","Error energy")):
            ax.set(xlabel="Cycles per grid interval",ylabel="Depth-mean spectral energy",title="Day 100 "+title);ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        for extension in ("png","pdf","svg"):fig.savefig(output/("day100_spectra."+extension),dpi=180)
        plt.close(fig)
    return result

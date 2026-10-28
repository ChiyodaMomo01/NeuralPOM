#!/usr/bin/env python3
"""Locked sequential eight-GPU suite: four training pairs, then all ten evaluations."""
import argparse, fcntl, json, os, subprocess, sys, time
from pathlib import Path
SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE/"src"))
from neuralpom.ai_baselines.config import load_config
from neuralpom.evaluation.protocol import atomic_json, sha256, source_hashes


def gpu_idle():
    output=subprocess.check_output(["nvidia-smi","-i","0,1,2,3,4,5,6,7", "--query-compute-apps=pid", "--format=csv,noheader"],text=True)
    return not output.strip()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",required=True);parser.add_argument("--resume",action="store_true")
    parser.add_argument("--tracking",choices=("auto","online","offline"),default="auto")
    args=parser.parse_args();path=Path(args.manifest).resolve();manifest=json.loads(path.read_text())
    root=Path(manifest["suite_root"]);root.mkdir(parents=True,exist_ok=True)
    lock=(root/"suite.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    statepath=root/"suite_state.json"
    identity=dict(manifest_sha256=sha256(path),source_sha256=source_hashes(SOURCE/"src/neuralpom"),
                  scripts_sha256=source_hashes(SOURCE/"scripts"))
    for entry in manifest["models"]:
        if sha256(entry["config"]) != entry["config_sha256"]:
            raise ValueError("Approved model configuration changed")
    configs={m["architecture"]:load_config(m["config"]) for m in manifest["models"]}
    if set(configs)!={"triton","resunet","simvp","convlstm","pastnet"}:raise ValueError("Five-model suite required")
    for cfg in configs.values():
        if cfg["world_size"]!=8:raise ValueError("Eight-rank suite required")
        if sha256(cfg["state_stats"])!=manifest["normalization_sha256"]:raise ValueError("Normalization differs")
    tasks=[]
    for m in manifest["models"]:
        if m["architecture"]!="triton":tasks.append(dict(action="training_pair",**m))
    for m in manifest["models"]:
        for stage in ("pretrain","posttrain"):tasks.append(dict(action="evaluate",stage=stage,**m))
    state=dict(schema="neuralpom-ai-suite-v1",**identity,pid=os.getpid(),status="running",started_at=time.time(),completed_tasks=[],tasks=tasks)
    if statepath.exists():
        if not args.resume:raise FileExistsError("Existing suite requires --resume")
        old=json.loads(statepath.read_text())
        if any(old[k]!=v for k,v in identity.items()):raise ValueError("Suite source or manifest changed")
        state["completed_tasks"]=old["completed_tasks"]
    logs=root/"logs";logs.mkdir(exist_ok=True)
    env=dict(os.environ,NEURALPOM_PYTHON=sys.executable,PYTHONUNBUFFERED="1")
    try:
        for index,task in enumerate(tasks):
            if index in state["completed_tasks"]:continue
            while not gpu_idle():
                state.update(status="waiting_for_free_gpus",current_task=index,updated_at=time.time());atomic_json(statepath,state)
                time.sleep(30)
            cfg=configs[task["architecture"]];run=Path(cfg["run_root"])
            if task["action"]=="training_pair":
                command=[sys.executable,str(SOURCE/"scripts/run_ai_stages.py"),"--config",task["config"],"--tracking",args.tracking]
                if (run/"pipeline.json").exists():command.append("--resume")
            else:
                # The entire training matrix is required before formal evaluation.
                for other in configs.values():
                    for stage in ("pretrain","posttrain"):
                        r=Path(other["run_root"])
                        if not (r/"frozen"/(stage+".pt")).exists() or not (r/stage/"COMPLETE.json").exists():
                            raise ValueError("Incomplete training matrix; evaluation blocked")
                command=["bash",str(SOURCE/"scripts/launch_ai_daily.sh"),"evaluate","--config",task["config"],"--stage",task["stage"],"--days","100"]
            name=task["architecture"]+"_"+task.get("stage",task["action"])
            state.update(status="running",current_task=index,current_name=name,updated_at=time.time());atomic_json(statepath,state)
            with (logs/(name+".log")).open("a") as stream:
                child=subprocess.Popen(command,cwd=SOURCE,env=env,stdout=stream,stderr=subprocess.STDOUT)
                state.update(worker_pid=child.pid);atomic_json(statepath,state);code=child.wait()
            if code:raise RuntimeError(name+" exited "+str(code))
            if task["action"]=="evaluate":
                summary=run/"evaluation"/(task["stage"]+"_100d")/"summary.json"
                if not summary.exists():raise ValueError("Missing formal evaluation summary")
                value=json.loads(summary.read_text())
                if len(value["statuses"])!=16 or any(s["status"] not in ("complete","numerical_failure") for s in value["statuses"]):
                    raise ValueError("Evaluation has unresolved cases")
            state["completed_tasks"].append(index);atomic_json(statepath,state)
        from neuralpom.ai_baselines.suite_report import summarize_suite
        summarize_suite(path)
        state.update(status="complete",completed_at=time.time())
    except Exception as error:
        state.update(status="failed",error=type(error).__name__+": "+str(error),failed_at=time.time());raise
    finally:atomic_json(statepath,state)

if __name__=="__main__":main()

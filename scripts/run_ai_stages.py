#!/usr/bin/env python3
"""Prepare once, then eight-GPU Pretrain followed by Posttrain on success."""
from __future__ import annotations
import argparse
import fcntl
import json
import netrc
import os
from pathlib import Path
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE/"src"))
from neuralpom.ai_baselines.config import load_config, plan
from neuralpom.evaluation.protocol import atomic_json, sha256, source_hashes


def has_wandb_credentials():
    if os.environ.get("WANDB_API_KEY"): return True
    try: return bool(netrc.netrc().authenticators("api.wandb.ai"))
    except (OSError, netrc.NetrcParseError): return False


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tracking", choices=("auto","online","offline"), default="auto")
    parser.add_argument("--resume", action="store_true")
    args=parser.parse_args()
    config=load_config(args.config)
    if config["world_size"] != 8 or config["model"]["architecture"] not in ("triton", "resunet", "simvp", "convlstm", "pastnet"):
        raise ValueError("This launch requires eight-rank registered AI models")
    root=Path(config["run_root"]);root.mkdir(parents=True, exist_ok=True)
    lock=(root/"pipeline.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=root/"pipeline.json"
    if state_path.exists() and not args.resume: raise FileExistsError("Existing pipeline requires --resume")
    state=dict(schema="neuralpom-ai-stage-pipeline-v1", pid=os.getpid(), source=str(SOURCE),
        config_sha256=sha256(args.config), source_sha256=source_hashes(SOURCE/"src/neuralpom"),
        stages=["prepare","pretrain","posttrain"], gpus=list(range(8)), plan=plan(config),
        started_at=time.time(), status="running", completed_stages=[])
    if args.resume:
        old=json.loads(state_path.read_text())
        for key in ("config_sha256","source_sha256"):
            if old[key] != state[key]: raise ValueError("Pipeline source/config changed")
        state["completed_stages"]=old["completed_stages"]
    if "prepare" not in state["completed_stages"] and Path(config["state_stats"]).exists():
        from neuralpom.ai_baselines.runtime import AIRuntime
        import torch
        AIRuntime(config, torch.device("cpu"))  # Verify immutable shared training-only statistics.
        state["completed_stages"].append("prepare")
    logs=root/"logs";logs.mkdir(exist_ok=True)
    env=dict(os.environ,NEURALPOM_PYTHON=sys.executable,PYTHONUNBUFFERED="1")
    try:
        for stage in state["stages"]:
            if stage in state["completed_stages"]: continue
            mode=args.tracking
            if mode=="auto": mode="online" if has_wandb_credentials() else "offline"
            if mode=="online" and not has_wandb_credentials(): raise RuntimeError("W&B credentials missing")
            effective=dict(config,wandb=dict(config["wandb"],mode=mode))
            path=root/f"{stage}_launch.json";atomic_json(path,effective)
            command=["bash",str(SOURCE/"scripts/launch_ai_daily.sh"),"prepare" if stage=="prepare" else "train","--config",str(path)]
            if stage!="prepare":
                command.extend(["--stage",stage])
                if (root/stage/"resume.json").exists(): command.append("--resume")
            if stage=="posttrain" and (not (root/"pretrain/COMPLETE.json").exists() or not (root/"frozen/pretrain.pt").exists()):
                raise ValueError("Pretrain did not complete with a frozen parent")
            state.update(current_stage=stage,tracking_mode=mode,updated_at=time.time());atomic_json(state_path,state)
            print(f"Starting {stage}; W&B {mode}",flush=True)
            with (logs/f"{stage}.log").open("a") as stream:
                child=subprocess.Popen(command,cwd=SOURCE,env=env,stdout=stream,stderr=subprocess.STDOUT)
                state.update(worker_pid=child.pid);atomic_json(state_path,state)
                code=child.wait()
            if code: raise RuntimeError(f"{stage} exited {code}; later stages were not started")
            if stage=="prepare":
                if not Path(config["state_stats"]).exists(): raise RuntimeError("Missing complete normalization")
            elif not (root/stage/"COMPLETE.json").exists() or not (root/f"frozen/{stage}.pt").exists():
                raise RuntimeError(f"{stage} omitted its completion/frozen checkpoint")
            state["completed_stages"].append(stage);atomic_json(state_path,state)
            print(f"Completed {stage}",flush=True)
        state.update(status="complete",completed_at=time.time())
    except Exception as error:
        state.update(status="failed",error=f"{type(error).__name__}: {error}",failed_at=time.time());raise
    finally: atomic_json(state_path,state)


if __name__=="__main__": main()

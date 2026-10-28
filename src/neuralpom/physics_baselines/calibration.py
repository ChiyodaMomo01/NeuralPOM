"""Frozen B1 validation screening, using the existing committed-trajectory writer."""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,subprocess,sys,time
from neuralpom.v1.checkpoint import atomic_json
from neuralpom.v1.config import load_config
from neuralpom.evaluation.runner import run_case,execution_audit
from .runtime import SmagorinskyRuntime


def load(path):
    path=Path(path);p=json.loads(path.read_text());p['_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    if p['family']!='B1' or p['selection_split']!='validation':raise ValueError('Not a B1 validation protocol')
    if any(c['member']!=7 or c['start_day']<7000 or c['start_day']+max(p['horizons_days'])>8000 for c in p['cases']):
        raise ValueError('Calibration crosses validation bounds')
    source=Path(__file__).resolve().parents[2]
    expected=json.loads((path.parent/'source_hashes.json').read_text())
    for name,digest in expected.items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source changed: '+name)
    return p


def worker(args):
    p=load(args.protocol)
    root=Path(p['output_root']);root.mkdir(parents=True,exist_ok=True)
    with (root/f'worker{args.index}.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for candidate in p['candidates'][args.index::args.workers]:
            config=load_config(p['config'])
            runtime=SmagorinskyRuntime(config,args.device,cm=candidate['cm'],cc=candidate['cc'])
            variant=dict(id=candidate['id'],group_id=p['core']+'_1h',core=p['core'],frequency='1h',stage='physics',policy='physics')
            metadata=dict(weight_sha256=None,variant=variant,training_contract=runtime.contract(),
                          execution=execution_audit(runtime),parameters=candidate,protocol_sha256=p['_sha256'])
            manifest=root/'runs'/variant['id']/'manifest.json'
            if manifest.exists() and json.loads(manifest.read_text())!=metadata:raise ValueError('Manifest mismatch')
            atomic_json(manifest,metadata)
            for case in p['cases']:
                print(json.dumps(dict(event='case_start',candidate=candidate,case=case,pid=os.getpid())),flush=True)
                status=run_case(p,variant,case,runtime,None,metadata,stop_after=args.days)
                print(json.dumps(dict(event='case_end',status=status)),flush=True)


def launch(args):
    p=load(args.protocol);root=Path(p['output_root']);root.mkdir(parents=True,exist_ok=True)
    with (root/'scheduler.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        jobs=[];handles=[]
        for index in range(args.workers):
            log=(root/f'worker{index}.log').open('a');handles.append(log)
            cmd=[sys.executable,'-m','neuralpom.physics_baselines.calibration','worker','--protocol',str(Path(args.protocol).resolve()),'--index',str(index),'--workers',str(args.workers),'--days',str(args.days),'--device',args.device]
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1'))
            jobs.append((process,cmd,index))
        while True:
            state=dict(pid=os.getpid(),protocol_sha256=p['_sha256'],days=args.days,updated_at=time.time(),
                       workers=[dict(index=i,pid=j.pid,returncode=j.poll(),command=c) for j,c,i in jobs])
            atomic_json(root/'scheduler.json',state)
            if all(j.poll() is not None for j,_,_ in jobs):break
            time.sleep(5)
        for h in handles:h.close()
        if any(j.returncode for j,_,_ in jobs):raise SystemExit(1)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['worker','launch']);p.add_argument('--protocol',required=True)
    p.add_argument('--index',type=int,default=0);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--days',type=int,default=10);p.add_argument('--device',default='cpu');args=p.parse_args()
    if args.workers<=0 or not 0<=args.index<args.workers:raise ValueError('Invalid worker assignment')
    (worker if args.action=='worker' else launch)(args)


if __name__=='__main__':main()

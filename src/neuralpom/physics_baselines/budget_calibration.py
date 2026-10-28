"""Frozen B2/B3 validation with parent-selection evidence and daily energy audits."""
import argparse,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from neuralpom.v1.checkpoint import atomic_json
from neuralpom.v1.config import load_config
from neuralpom.evaluation.runner import execution_audit
from .budget_split import BudgetRuntime
from .budget_case import run_budget_case


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    path=Path(path);p=json.loads(path.read_text());p['_sha256']=digest(path)
    if p.get('schema')!='physics-budget-calibration-v1' or p.get('family') not in ('B2','B3') or p.get('selection_split')!='validation':raise ValueError('Invalid budget calibration protocol')
    if p['horizons_days']!=[10,50,200]:raise ValueError('Unexpected validation horizons')
    if p['cases']!=[dict(member=7,start_day=d) for d in (7000,7240,7480,7720)]:raise ValueError('Unexpected validation origins')
    hashes=json.loads((path.parent/'source_hashes.json').read_text())
    actual={str(f.relative_to(Path(__file__).resolve().parents[2])) for f in Path(__file__).resolve().parents[2].rglob('*.py') if '__pycache__' not in f.parts}
    if set(hashes)!=actual:raise ValueError('Incomplete source manifest')
    for name,h in hashes.items():
        if digest(Path(__file__).resolve().parents[2]/name)!=h:raise ValueError('Frozen source changed: '+name)
    parent=p['parent_selection']
    if digest(parent['path'])!=parent['sha256']:raise ValueError('Parent selection evidence changed')
    selected=json.loads(Path(parent['path']).read_text())
    expected_parent='B1' if p['family']=='B2' else 'B2'
    if selected.get('status') not in ('selected_for_validation','frozen') or selected['family']!=expected_parent or selected['core']!=p['core']:raise ValueError('Parent selection not valid')
    params=selected['parameters']
    if params['cm']!=p['cm'] or params['cc']!=p['cc']:raise ValueError('Inherited coefficients differ')
    if p['family']=='B3' and params['gamma']!=p['gamma']:raise ValueError('B3 must inherit its B2 dissipation')
    evidence_roles={x['role'] for x in p['implementation_validation']}
    if evidence_roles!={'coupled_pilot','commit_explicit','commit_implicit_p0_fixed'} or len(p['implementation_validation'])!=3:raise ValueError('Need coupled pilot and both commit validations')
    for evidence in p['implementation_validation']:
        if digest(evidence['path'])!=evidence['sha256']:raise ValueError('Implementation evidence changed')
        value=json.loads(Path(evidence['path']).read_text())
        if value.get('status')!='passed':raise ValueError('Implementation gate not passed')
        if evidence['role'].startswith('commit_'):
            if value.get('core')!=evidence['role'][7:] or value.get('days')!=2 or not value.get('latest_budget_repair_exact'):raise ValueError('Wrong commit validation')
        elif {x['core'] for x in value['results']}!={'explicit','implicit_p0_fixed'}:raise ValueError('Incomplete coupled pilot')
    candidates=p['candidates']
    if len(candidates)!=3 or len({c['id'] for c in candidates})!=3:raise ValueError('Expected three unique candidates')
    if p['family']=='B2':
        if sorted(c['gamma'] for c in candidates)!=[.5,1.,2.] or any(c['beta']!=0 for c in candidates):raise ValueError('Unregistered B2 search')
    elif sorted(c['beta'] for c in candidates)!=[.5,.75,1.] or any(c['gamma']!=p['gamma'] for c in candidates):raise ValueError('Unregistered B3 search')
    return p


def worker(args):
    p=load(args.protocol);root=Path(p['output_root']);root.mkdir(parents=True,exist_ok=True)
    selected=args.candidates or [c['id'] for c in p['candidates']]
    with (root/f'worker{args.index}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for cid in selected[args.index::args.workers]:
            c=next(c for c in p['candidates'] if c['id']==cid)
            config=load_config(p['config'])
            if config['core']!=p['core'] or config['frequency']!='1h':raise ValueError('Core or reference frequency differs')
            r=BudgetRuntime(config,args.device,cm=p['cm'],cc=p['cc'],gamma=c['gamma'],beta=c['beta'],family=p['family'])
            v=dict(id=cid,group_id=p['core']+'_1h',core=p['core'],frequency='1h',stage='physics',policy='physics')
            metadata=dict(weight_sha256=None,variant=v,training_contract=r.contract(),execution=execution_audit(r),parameters=c,protocol_sha256=p['_sha256'],config_sha256=digest(p['config']))
            manifest=root/'runs'/cid/'manifest.json'
            if manifest.exists() and json.loads(manifest.read_text())!=metadata:raise ValueError('Manifest mismatch')
            atomic_json(manifest,metadata)
            for case in p['cases']:
                print(json.dumps(dict(event='case_start',candidate=cid,case=case)),flush=True)
                status=run_budget_case(p,v,case,r,metadata,stop_after=args.days)
                print(json.dumps(dict(event='case_end',status=status)),flush=True)


def launch(args):
    p=load(args.protocol);root=Path(p['output_root']);root.mkdir(parents=True,exist_ok=True)
    selected=args.candidates or [c['id'] for c in p['candidates']]
    with (root/'scheduler.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # A lost parent lock does not prove its workers have stopped.
        for oldfile in root.glob('scheduler*.json'):
            old=json.loads(oldfile.read_text())
            for item in [old,*old['workers']]:
                proc=Path(f"/proc/{item['pid']}/cmdline")
                command=proc.read_bytes().replace(b'\0',b' ').decode() if proc.exists() else ''
                if 'budget_calibration' in command and str(Path(args.protocol).resolve()) in command:
                    raise RuntimeError('Existing calibration process still live')
        jobs=[];logs=[]
        for index in range(args.workers):
            log=(root/f'worker{index}_{args.days}d.log').open('a');logs.append(log)
            cmd=[sys.executable,'-m',__name__ if __name__!='__main__' else 'neuralpom.physics_baselines.budget_calibration','worker','--protocol',str(Path(args.protocol).resolve()),'--index',str(index),'--workers',str(args.workers),'--days',str(args.days),'--device',args.device,'--candidates',*selected]
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1'))
            jobs.append((process,cmd,index))
        while True:
            state=dict(pid=os.getpid(),protocol_sha256=p['_sha256'],days=args.days,candidates=selected,updated_at=time.time(),workers=[dict(index=i,pid=j.pid,returncode=j.poll(),command=c) for j,c,i in jobs])
            atomic_json(root/f'scheduler{args.days}.json',state)
            if all(j.poll() is not None for j,_,_ in jobs):break
            time.sleep(5)
        for f in logs:f.close()
        if any(j.returncode for j,_,_ in jobs):raise SystemExit(1)


def main():
    a=argparse.ArgumentParser();a.add_argument('action',choices=['launch','worker']);a.add_argument('--protocol',required=True);a.add_argument('--workers',type=int,default=3);a.add_argument('--index',type=int,default=0);a.add_argument('--days',type=int,choices=[10,50,200],default=10);a.add_argument('--device',default='cpu');a.add_argument('--candidates',nargs='+');args=a.parse_args()
    if args.workers<1 or not 0<=args.index<args.workers:raise ValueError('Invalid worker assignment')
    p=load(args.protocol)
    if args.candidates and (len(set(args.candidates))!=len(args.candidates) or not set(args.candidates)<=set(c['id'] for c in p['candidates'])):raise ValueError('Unknown or duplicate candidate')
    (launch if args.action=='launch' else worker)(args)

if __name__=='__main__':main()

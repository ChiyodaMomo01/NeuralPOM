"""Existing trajectory commits plus independently restartable daily budget files."""
from pathlib import Path
import torch
from neuralpom.evaluation.runner import run_case
from neuralpom.evaluation.protocol import case_root,read_json,atomic_json
from .ledger import committed_budget


def run_budget_case(protocol,variant,case,runtime,metadata,*,stop_after):
    root=case_root(protocol['output_root'],variant['id'],case)
    status=read_json(root/'status.json') if (root/'status.json').exists() else {'completed_days':0,'status':'pending'}
    if status['status']=='infrastructure_error':raise ValueError('System error requires explicit reviewed recovery')
    target=min(int(stop_after),max(protocol['horizons_days']))
    if target<=0:raise ValueError('Positive target required')
    def audit_commit(day):
        saved=torch.load(root/'restart.pt',map_location='cpu',weights_only=False)
        if saved['day']!=day:raise ValueError('Budget and physical commit differ')
        expected=dict(variant=variant['id'],**case,target_days=max(protocol['horizons_days']),weight_sha256=None,protocol_sha256=protocol['_sha256'])
        if saved['identity']!=expected:raise ValueError('Budget provenance differs')
        result=committed_budget(saved['state'])
        if abs(result['time']-case['start_day']-day)>2e-6:raise ValueError('Budget clock differs')
        result.update(lead_day=day,identity=saved['identity'])
        path=root/'budgets'/f'day{day:03d}.json'
        if path.exists() and read_json(path)!=result:raise ValueError('Committed budget was changed')
        atomic_json(path,result)
    completed=status['completed_days']
    # An interruption after physical commit but before budget publication can
    # repair the latest day from restart. Earlier missing accounts are not invented.
    for day in range(1,completed):
        if not (root/'budgets'/f'day{day:03d}.json').exists():raise ValueError('Historical daily budget missing')
    if completed:audit_commit(completed)
    if status['status'] in ('complete','numerical_failure'):return status
    for day in range(completed+1,target+1):
        status=run_case(protocol,variant,case,runtime,None,metadata,stop_after=day)
        if status['completed_days']==day:audit_commit(day)
        if status['status'] in ('numerical_failure','infrastructure_error'):break
    return status

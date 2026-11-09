"""Report committed full-population progress without fitting or projecting."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from core import ROOT,BRANCHES,load_plan,verify_code,paths,save_json


def main():
    sha=verify_code(); plan=load_plan(); audit=json.loads((ROOT/'input_audit.json').read_text())
    assert audit['passed'] and audit['manifest_sha256']==sha
    totals={b:Counter() for b in BRANCHES}; incomplete=[]; errors=[]
    for task in plan['tasks']:
        final,checkpoint=paths(task); path=final if final.exists() else checkpoint if checkpoint.exists() else None
        n=task['stop']-task['start']; b=task['branch']
        totals[b]['target']+=n
        if path is None:
            totals[b]['pending']+=n; incomplete.append(task['id']); continue
        try:
            with np.load(path,allow_pickle=False) as f:
                meta=json.loads(str(f['metadata']))
                assert meta['task']==task and meta['manifest_sha256']==sha
                assert meta['input_file_sha256']==audit['cache_sha256'][plan['parts'][task['part']]['path']]
                status=f['status']; assert status.shape==(n,)
                assert np.all(np.isin(status,[0,1,-1,-2]))
                if path==final: assert bool(f['complete']) and np.all(status!=0)
                for state,key in [(0,'pending'),(1,'success'),(-1,'failed'),(-2,'timeout')]: totals[b][key]+=int(np.sum(status==state))
                totals[b]['quality_flagged_success']+=int(np.sum((status==1)&(f['quality_flags']!=0)))
                if np.any(status==0): incomplete.append(task['id'])
        except Exception as exc:
            errors.append(dict(task=task['id'],path=str(path),error=f'{type(exc).__name__}: {exc}'))
            incomplete.append(task['id']); totals[b]['unreadable']+=n
    lane_status={}
    for lane in range(4):
        path=ROOT/f'lanes/lane{lane:02d}.json'
        lane_status[str(lane)]=json.loads(path.read_text()) if path.exists() else None
    result=dict(complete=not incomplete and not errors,manifest_sha256=sha,physical_state_target=plan['physical_states'],
        totals={b:dict(v) for b,v in totals.items()},incomplete_blocks=len(incomplete),
        incomplete_task_ids=incomplete,corrupt_or_mismatched_results=errors,lane_status=lane_status,
        note='Failed/timeout rows are attempted but remain NaN; no imputation, automatic retry, model training or scientific acceptance.')
    save_json(ROOT/'progress.json',result)
    text='# Full snapshot575 x8 direct calculation: progress\n\n'
    text+='Complete population attempted: '+str(result['complete'])+'\n\n'
    text+='| Branch | Target | Success | Failed | Timeout | Pending |\n|---|---:|---:|---:|---:|---:|\n'
    for b,v in totals.items(): text+=f"| {b} | {v['target']} | {v['success']} | {v['failed']} | {v['timeout']} | {v['pending']} |\n"
    text+='\nIf pending states remain, resume the same lanes after reviewing resource use. No data or thresholds need changing.\n'
    (ROOT/'PROGRESS.md').write_text(text)
    print(text,flush=True)


if __name__=='__main__': main()

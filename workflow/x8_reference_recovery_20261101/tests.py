"""Check observational wrappers and time-horizon forwarding with stubs only."""
from collections import Counter
import copy
from types import SimpleNamespace

import numpy as np
import shared as s
import run_cases as r


def main():
    r.EVENTS=Counter();r.FINAL={};r.FINAL_DAMPING={};r.PHASE='chemistry'
    emitter=SimpleNamespace(name='C18O',dampFactor=.5)
    failure=r.wrap_emitter(lambda *args,**kwargs:False)
    success=r.wrap_emitter(lambda *args,**kwargs:True)
    assert failure(emitter) is False and not r.FINAL
    r.PHASE='capture';assert success(emitter) is True
    assert r.FINAL['c18o']==1 and r.FINAL_DAMPING['c18o']==.5
    assert failure(emitter,dampFactor=.03) is False
    assert r.FINAL['c18o']==0 and r.FINAL_DAMPING['c18o']==.03
    assert r.EVENTS[('chemistry','c18o','fail')]==1
    def broken(model):raise ValueError('expected test')
    try:r.wrap_capture(broken)(None)
    except ValueError:pass
    else:raise AssertionError('Capture wrapper swallowed exception')
    assert r.PHASE=='chemistry'
    original=r.worker.DIRECT,r.worker.solve,r.BASE_SOLVER
    solver=dict(tEqGuess_s=None,maximum_time_primary_s=1e16,maximum_time_retry_s=1e18,relative_tolerance=.001)
    r.worker.DIRECT=SimpleNamespace(SOLVER=copy.deepcopy(solver));r.BASE_SOLVER=solver
    calls=[]
    def fake(task):
        calls.append(copy.deepcopy(r.worker.DIRECT.SOLVER))
        return task[0],1,1,.1,1,0,0,np.ones(27),np.ones(20),''
    r.worker.solve=fake
    try:
        r.solve((0,0,'actual_on',(1,2,3,4,5,6,7)))
        r.solve((1,1,'actual_on',(1,2,3,4,5,6,7)))
        r.solve((2,2,'actual_on',(1,2,3,4,5,6,7)))
    finally:r.worker.DIRECT,r.worker.solve,r.BASE_SOLVER=original
    assert [x['maximum_time_primary_s'] for x in calls]==[1e18,1e16,1e16]
    assert all(x['tEqGuess_s'] is None and x['relative_tolerance']==.001 for x in calls)
    s.save(s.ROOT/'local_checks.json',dict(passed=True,transparent_emitter_returns=True,
        phase_separation=True,last_final_call_recorded=True,capture_exception_preserved=True,
        AUTO_preserved=True,retry_horizon_only_override=True,replay_horizon_unchanged=True))
    print('Phase wrappers and AUTO/time-horizon forwarding checks passed; no chemistry executed.')


if __name__=='__main__':main()

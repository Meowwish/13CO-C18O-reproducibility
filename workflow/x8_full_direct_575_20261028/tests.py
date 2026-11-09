"""Deterministic correctness checks; never run physical chemistry here."""
from concurrent.futures import Future
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import core
import engine
import worker


def run():
    registry=core.load_plan()
    totals={b:0 for b in core.BRANCHES}
    for b in core.BRANCHES:
        for part in range(8):
            entries=[x for x in registry['tasks'] if x['branch']==b and x['part']==part]
            assert entries[0]['start']==0 and entries[-1]['stop']==registry['parts'][part]['count']
            for a,z in zip(entries,entries[1:]): assert a['stop']==z['start']
            assert all(x['stop']>x['start'] for x in entries)
            totals[b]+=sum(x['stop']-x['start'] for x in entries)
    assert set(totals.values())=={43133718}
    assert [x['id'] for x in registry['tasks']]==list(range(len(registry['tasks'])))
    lanes=[{x['id'] for x in registry['tasks'] if x['id']%4==lane} for lane in range(4)]
    assert sum(map(len,lanes))==len(set.union(*lanes))==len(registry['tasks'])
    assert len(core.SPECIES)==20 and len(core.OUTPUT_NAMES)==27
    xyz=np.array([[100.,200.,0.],[101.,201.,20.],[1100.,200.,0.]])
    assert core.split_group(xyz)[0]==core.split_group(xyz)[1]
    assert np.array_equal(core.split_group(xyz),core.split_group(xyz.copy()))
    # The model is a stub: inspect exactly what the production call forwards.
    calls=[]; damping=[]
    original=(worker.DIRECT,worker.GOW,worker.capture,worker.ORIGINAL_ODEINT)
    solver=dict(tEqGuess_s=None,evolve_temperature='iterate',add_emitters=True,
        maximum_time_primary_s=1e16,relative_tolerance=.001,convergence_species=None,maximum_temperature_iterations=50)
    def chem(**kw): calls.append(kw); worker.COUNTS['calls']+=1; return True
    def make(row,damp): damping.append(damp); return SimpleNamespace(setDustTempEq=lambda:True,setChemEq=chem)
    fake=SimpleNamespace(CHEMISTRY={},CONFIG=dict(execution=dict(primary=dict(damp_factors=[.25,.1,.03]))),
        SOLVER=solver,CLOUD=dict(no_clumping=True,escape_probability_geometry='LVG'),SMALL_ABUNDANCE=1e-15,make_cloud=make)
    worker.DIRECT=fake; worker.GOW=object
    worker.capture=lambda model:(np.ones(27),np.ones(20),0)
    import signal
    previous=signal.signal(signal.SIGALRM,worker.alarm_handler)
    try:
        result=worker.solve((0,'actual_on',(1,2,3,4,5,6,7)))
        assert result[1]==1 and calls[0]['tEqGuess'] is None
        assert calls[0]['smallabd']==1e-15 and calls[0]['tol']==.001
        assert damping==[.25] and fake.CHEMISTRY['include_13C_fractionation']
        def fail_first(row,damp):
            if damp==.25: raise RuntimeError('Controlled first-attempt failure')
            return make(row,damp)
        fake.make_cloud=fail_first
        result=worker.solve((0,'actual_off',(1,2,3,4,5,6,7)))
        assert result[1:3]==(1,2) and not fake.CHEMISTRY['include_13C_fractionation']
        fake.make_cloud=lambda *args:(_ for _ in ()).throw(RuntimeError('Controlled all-attempt failure'))
        failed=worker.solve((0,'actual_on',(1,2,3,4,5,6,7)))
        assert failed[1:3]==(-1,3) and np.isnan(failed[7]).all()
        def timeout(*args): raise worker.CaseTimeout('Controlled timeout')
        fake.make_cloud=timeout
        timed=worker.solve((0,'actual_on',(1,2,3,4,5,6,7)))
        assert timed[1]==-2 and np.isnan(timed[7]).all()
        worker.COUNTS=dict(calls=0,failures=0)
        def failed_ode(*args,**kwargs):
            assert 'mxstep' not in kwargs
            return np.zeros((2,2)),dict(message='Excess work done')
        worker.ORIGINAL_ODEINT=failed_ode
        rejected=False
        try: worker.checked_odeint([])
        except RuntimeError: rejected=True
        assert rejected and worker.COUNTS['failures']==1
    finally:
        worker.DIRECT,worker.GOW,worker.capture,worker.ORIGINAL_ODEINT=original
        signal.signal(signal.SIGALRM,previous)
    # Store the actual emitter coefficient, not an assumed tau transform.
    old_direct=worker.DIRECT
    try:
        worker.DIRECT=SimpleNamespace(line_diagnostics=lambda *args:(1e-27,1.,4.,.01))
        beta=[.7,.8,.9]
        emitters={name:SimpleNamespace(escapeProb=np.array([[1.,b],[b,1.]])) for name,b in zip(('co','13co','c18o'),beta)}
        model=SimpleNamespace(Tg=8.,Td=10.,emitters=emitters,
            comp=SimpleNamespace(xHI=.1,xHplus=1e-4,xe=1e-4,xpH2=.3,xoH2=.15,xHe=.1),
            chemnetwork=SimpleNamespace(abundances={'H2':.45,'CO':1e-4,'13CO':1e-6,'C18O':1e-7},x=np.ones(20)*1e-7))
        values,chemistry,flags=worker.capture(model)
        np.testing.assert_allclose(values[18:21],beta)
        assert values.shape==(27,) and chemistry.shape==(20,)
    finally: worker.DIRECT=old_direct
    data=dict(row=np.array([3,7]),particle_id=np.array([10,20]),split=np.array([0,2]))
    task=dict(id=0,part=0,branch='actual_on',start=0,stop=2)
    metadata=dict(task=task,manifest_sha256='stub',input_file_sha256='stub',block_input_sha256='stub')
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'checkpoint.npz'; state=core.empty_state(2)
        success=(0,1,1,.1,1,0,0,np.ones(27),np.ones(20),'')
        engine.apply_result(state,success)
        core.atomic_state(p,metadata,state,data,False)
        resumed=core.read_state(p,metadata,2)
        np.testing.assert_array_equal(resumed['status'],[1,0])
        assert np.isnan(resumed['values'][1]).all()
        bad=copy.deepcopy(metadata); bad['manifest_sha256']='wrong'
        rejected=False
        try: core.read_state(p,bad,2)
        except AssertionError: rejected=True
        assert rejected
        engine.apply_result(resumed,(1,-1,3,.2,2,1,0,np.full(27,np.nan),np.full(20,np.nan),'failed'))
        core.atomic_state(p,metadata,resumed,data,True)
        complete=core.read_state(p,metadata,2,require_complete=True)
        assert complete['status'].tolist()==[1,-1] and np.isnan(complete['values'][1]).all()
        overwritten=False
        try: engine.apply_result(complete,success)
        except AssertionError: overwritten=True
        assert overwritten
        p.write_bytes(b'corrupted-checkpoint')
        rejected=False
        try: core.read_state(p,metadata,2)
        except (ValueError,OSError): rejected=True
        assert rejected
    return dict(full_population_per_branch=totals,lane_partition=True,auto_forwarding=True,
        tolerance_forwarding=True,damping_order=True,ODE_failure_rejection=True,no_mxstep_override=True,
        timeout_policy=True,NaN_failure_values=True,spatial_split_grouping=True,
        atomic_checkpoint_resume=True,identity_mismatch_rejection=True,corrupt_checkpoint_rejection=True,
        actual_beta_extraction=True,no_committed_overwrite=True)


if __name__=='__main__': print(json.dumps(run(),indent=2))

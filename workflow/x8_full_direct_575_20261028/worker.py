"""One primary auto-time DESPOTIC solve; no physical solver runs on import."""
import importlib
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
from core import ROOT,OUTPUT_NAMES,SPECIES,digest

DIRECT=None
GOW=None
ORIGINAL_ODEINT=None
COUNTS=None
POP_FAILURES=0


class CaseTimeout(BaseException): pass


def alarm_handler(*_):
    raise CaseTimeout('180 s total physical-state wall budget exceeded')


def checked_odeint(*args,**kwargs):
    requested=kwargs.pop('full_output',False)
    # Preserve the primary library work limit: no mxstep override.
    COUNTS['calls']+=1
    values,info=ORIGINAL_ODEINT(*args,full_output=True,**kwargs)
    if info.get('message')!='Integration successful.' or not np.all(np.isfinite(values)):
        COUNTS['failures']+=1
        raise RuntimeError('Rejected ODE: '+str(info.get('message')))
    return (values,info) if requested else values


def initialize(quiet=True,staged=True):
    global DIRECT,GOW,ORIGINAL_ODEINT
    import solve_direct_chunk as direct
    from despotic.chemistry.GOW import GOW as network,specList
    DIRECT=direct; GOW=network
    assert direct.SOLVER['tEqGuess_s'] is None and direct.SMALL_ABUNDANCE==1e-15
    assert direct.SOLVER['relative_tolerance']==.001
    assert tuple(direct.CONFIG['execution']['primary']['damp_factors'])==(.25,.1,.03)
    assert tuple(specList)==SPECIES and len(OUTPUT_NAMES)==27
    if staged:
        stage=Path(os.environ['FULL575_STAGE'])
        assert digest(stage/'MilkyWayGMC.desp')==digest(direct.CLOUD_FILE)
        direct.CLOUD_FILE=stage/'MilkyWayGMC.desp'
        os.environ['VISSER_CO_SHIELDING_DIR']=str(stage/'visser')
    else:
        os.environ['DESPOTIC_HOME']=str(direct.ROOT)
    module=importlib.import_module('despotic.chemistry.chemEvol')
    assert module.odeint is not checked_odeint,'Worker initialized twice'
    ORIGINAL_ODEINT=module.odeint
    module.odeint=checked_odeint
    # Preserve numerical behavior while exposing any internally reported failure.
    em=importlib.import_module('despotic.emitter').emitter
    original=em.setLevPopEscapeProb
    def observe(self,*args,**kwargs):
        global POP_FAILURES
        result=original(self,*args,**kwargs)
        if result is False: POP_FAILURES+=1
        return result
    em.setLevPopEscapeProb=observe
    signal.signal(signal.SIGALRM,alarm_handler)
    if quiet:
        # LSODA writes to Fortran stdout; retain machine-readable errors instead.
        with open(os.devnull,'w') as f:
            os.dup2(f.fileno(),1); os.dup2(f.fileno(),2)


def capture(model):
    abundance=model.chemnetwork.abundances
    molecular=np.array([float(abundance[k]) for k in ('H2','CO','13CO','C18O')])
    if np.any(~np.isfinite(molecular)) or np.any(molecular < -1e-12): raise RuntimeError('Nonphysical molecular abundance')
    lines=[]; betas=[]
    for species in ('co','13co','c18o'):
        lines.append(DIRECT.line_diagnostics(model,species))
        betas.append(float(model.emitters[species].escapeProb[1,0]))
    composition=[float(getattr(model.comp,k)) for k in ('xHI','xHplus','xe','xpH2','xoH2','xHe')]
    values=np.array([model.Tg,model.Td,*np.maximum(molecular,0),
        *[r[0] for r in lines],*[r[1] for r in lines],*[r[2] for r in lines],*[r[3] for r in lines],
        *betas,*composition],dtype=float)
    chemistry=np.asarray(model.chemnetwork.x,dtype=float).copy()
    if np.any(~np.isfinite(values)) or np.any(~np.isfinite(chemistry)) or model.Tg<=0:
        raise RuntimeError('Nonfinite direct output or nonpositive gas temperature')
    flags=int(np.any(values[9:12]<0))+2*int(np.any(values[12:15]<=0))+4*int(np.any(values[6:9]<0))
    flags+=8*int(POP_FAILURES>0)+16*int(np.any(molecular<0))
    return values,chemistry,flags


def solve(task):
    global COUNTS,POP_FAILURES
    index,branch,row=task
    COUNTS=dict(calls=0,failures=0); POP_FAILURES=0
    DIRECT.CHEMISTRY['include_13C_fractionation']=branch.endswith('_on')
    started=time.monotonic(); state=-1; attempts=0; error=''; values=np.full(len(OUTPUT_NAMES),np.nan); chemistry=np.full(len(SPECIES),np.nan); flags=0
    signal.alarm(180)
    try:
        for damp in DIRECT.CONFIG['execution']['primary']['damp_factors']:
            attempts+=1
            try:
                model=DIRECT.make_cloud(tuple(row),float(damp)); model.setDustTempEq()
                ok=model.setChemEq(tEqGuess=DIRECT.SOLVER['tEqGuess_s'],network=GOW,
                    evolveTemp=DIRECT.SOLVER['evolve_temperature'],
                    tempEqParam=dict(noClump=DIRECT.CLOUD['no_clumping'],escapeProbGeom=DIRECT.CLOUD['escape_probability_geometry']),
                    addEmitters=DIRECT.SOLVER['add_emitters'],maxTime=float(DIRECT.SOLVER['maximum_time_primary_s']),
                    tol=DIRECT.SOLVER['relative_tolerance'],smallabd=DIRECT.SMALL_ABUNDANCE,
                    convList=DIRECT.SOLVER['convergence_species'],maxTempIter=DIRECT.SOLVER['maximum_temperature_iterations'])
                if not ok: raise RuntimeError('Chemical/thermal convergence flag false')
                values,chemistry,flags=capture(model)
                assert COUNTS['calls']>0
                state=1; error=''; break
            except CaseTimeout: raise
            except Exception as exc: error=f'{type(exc).__name__}: {exc}'
    except CaseTimeout as exc:
        state=-2; error=f'CaseTimeout: {exc}'
    finally: signal.alarm(0)
    if state!=1:
        values[:]=np.nan; chemistry[:]=np.nan
    return (index,state,attempts,time.monotonic()-started,COUNTS['calls'],COUNTS['failures'],flags,values,chemistry,error[:760])

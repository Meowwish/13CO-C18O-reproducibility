"""Keep primary, replay-qualified and one-retry references separately traceable."""
from collections import Counter
import json
import os

import h5py
import numpy as np
import shared as s
from review_sample import assess,PAIRS


def main():
    assert int(os.environ.get('PBS_NCPUS','0'))>0
    sha=s.verify();target=s.ROOT/'review.json';assert not target.exists()
    with h5py.File(s.ROOT/'cases.h5') as f:
        ids=f['state_id'][:];modes=f['mode'][:];primary_values=f['primary_values'][:]
    ncases=len(ids);result={k:v for k,v in s.base.core.empty_state(ncases).items()}
    final=np.full((ncases,3),-1,'i1');seen=np.zeros(ncases,bool);event_counts=Counter()
    for part in range(8):
        with np.load(s.ROOT/f'output/part{part:02d}.npz',allow_pickle=False) as f:
            assert bool(f['complete']) and json.loads(str(f['metadata']))['manifest_sha256']==sha
            case=f['case_ids'];assert not seen[case].any() and np.all(case%8==part)
            np.testing.assert_array_equal(f['state_id'],ids[case])
            for key in result:result[key][case]=f[key]
            final[case]=f['final_returns'];seen[case]=True
            for raw in f['events']:event_counts.update(json.loads(raw.decode()))
    assert seen.all()
    nominal=result['status']==1;final_ok=np.all(final==1,axis=1);admissible=nominal&final_ok
    old=primary_values[:,6:9];new=result['values'][:,6:9]
    both_positive=(old>0)&(new>0)
    relative=np.abs(new-old)/np.maximum(np.maximum(np.abs(new),np.abs(old)),1e-60)
    with np.errstate(invalid='ignore',divide='ignore'):
        delta=np.abs(np.log10(new/old))
    agree=np.all(np.where(both_positive,delta<=1e-5,relative<=1e-5),axis=1)&nominal
    summary={}
    for mode,name in ((0,'failed_state_retry'),(1,'flagged_primary_replay'),(2,'unflagged_control')):
        pick=modes==mode
        summary[name]=dict(cases=int(pick.sum()),nominal_success=int(np.sum(pick&nominal)),
            final_emitter_success=int(np.sum(pick&admissible)),
            final_failure_despite_nominal_success=int(np.sum(pick&nominal&~final_ok)),
            spectrum_reproduced=int(np.sum(pick&agree)),
            final_ok_and_reproduced=int(np.sum(pick&admissible&agree)))
    base=s.base
    with h5py.File(s.SOURCE/'sample.h5') as f:
        n=len(f['sample/mass']);data={k:f['sample/'+k][:] for k in ('mass','nuclei','xyz','inclusion_probability','stratum')}
        proxy=f['sample/lum_per_H'][:];pop=f['stratum_population'][:];sn=f['stratum_sample'][:]
        exact={g:f['predicted_'+g+'_totals'][:] for g in ('particle','beam')}
    original_values=np.full((n*5,27),np.nan);original_status=np.zeros(n*5,'i1');flags=np.zeros(n*5,'u2')
    for part in range(8):
        with np.load(s.SOURCE/f'direct/part{part:02d}.npz',allow_pickle=False) as f:
            tid=f['task_ids'];original_values[tid]=f['values'];original_status[tid]=f['status'];flags[tid]=f['quality_flags']
    qualified=(original_status==1)&((flags&8)==0)
    verified=(modes==1)&admissible&agree
    qualified[ids[verified]]=True
    recovered=(modes==0)&admissible
    candidate_values=original_values.copy();candidate_values[ids[recovered]]=result['values'][recovered]
    qualified[ids[recovered]]=True
    origin=np.zeros(n*5,'u1');origin[(original_status==1)&((flags&8)==0)]=1
    origin[ids[verified]]=2;origin[ids[recovered]]=3
    base.atomic_npz(s.ROOT/'candidate_overlay.npz',manifest_sha256=np.asarray(sha),
        values=candidate_values.reshape(n,5,27),qualified=qualified.reshape(n,5),origin=origin.reshape(n,5),
        source_state_id=ids,recovery_status=result['status'],final_returns=final,
        replay_agreement=agree)
    kernel=np.full((n,9),np.nan)
    for part in range(4):
        with np.load(s.SOURCE/f'geometry/part{part:02d}.npz',allow_pickle=False) as f:kernel[f['ids']]=f['kernel_weights']
    frequencies=json.loads((s.ROOT.parent/'highres_snapshot575_temporal/config.json').read_text())['projection']['line_frequencies_Hz']
    freq=3*np.log10(frequencies['C18O']/frequencies['13CO'])
    direct=candidate_values.reshape(n,5,27)[:,base.MODEL_BRANCH,6:9]
    valid=qualified.reshape(n,5)[:,base.MODEL_BRANCH]
    records=[]
    for geometry,geometry_weight in (('particle',base.region_masks(data['xyz']).astype(float)),('beam',kernel)):
        for r,region in enumerate(base.REGIONS):
            weight=data['nuclei']/data['inclusion_probability']*geometry_weight[:,r]
            mass=data['mass']/data['inclusion_probability']*geometry_weight[:,r]
            for model in range(6):
                value=assess([model],direct,proxy,valid,weight,mass,data['stratum'],pop,sn,exact[geometry][r],freq)
                records.append(dict(geometry=geometry,region=region,comparison=base.MODELS[model],**value))
            for comparison,models in PAIRS.items():
                value=assess(list(models),direct,proxy,valid,weight,mass,data['stratum'],pop,sn,exact[geometry][r],freq)
                records.append(dict(geometry=geometry,region=region,comparison=comparison,**value))
    output=dict(complete=True,manifest_sha256=sha,case_summary=summary,
        original_failed_states=int(np.sum(original_status<0)),newly_recovered_failed_states=int(recovered.sum()),
        unresolved_or_unverified_states=int(np.sum(~qualified)),phase_event_counts=dict(event_counts),
        results=records,manuscript_acceptance=False,
        note='Candidate overlay only: source primary labels are immutable. Retry horizon is 1e18 s with AUTO guess; flagged replays use unchanged 1e16 s primary. Unknown references retain explicit masks.')
    s.save(target,output)
    text=['# Reference recovery and final-emitter audit','',
        '| Case class | Cases | Nominal success | Final emitters all succeed | Final fail despite nominal success |',
        '|---|---:|---:|---:|---:|']
    for name,d in summary.items():text.append(f"| {name} | {d['cases']} | {d['nominal_success']} | {d['final_emitter_success']} | {d['final_failure_despite_nominal_success']} |")
    text+=['',f"Recovered primary failures: {recovered.sum()}. Unresolved/unverified candidate states: {np.sum(~qualified)}.",'',
        'Primary labels remain unchanged. `review.json` reports phase distinctions and survey estimates on the qualified candidate overlay. No automatic manuscript acceptance.','']
    (s.ROOT/'RESULTS.md').write_text('\n'.join(text))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()

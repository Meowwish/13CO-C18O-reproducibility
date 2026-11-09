"""Two streaming passes; sampling never reads direct labels."""
import json
import os
from pathlib import Path
import h5py
import numpy as np

import common as c

FIELDS=('actual','slug','mass','xyz','h','row','particle_id','nuclei','lum_per_H','covered')


def chunks(path):
    with h5py.File(path) as f:
        assert f.attrs['complete']
        for start in range(0,len(f['mass']),100000):
            data={k:f[k][start:start+100000] for k in FIELDS}
            data['cache_row']=np.arange(start,start+len(data['mass']),dtype='i8')
            yield data


def reuse_actual(selected, plan, audit):
    n=len(selected['mass']);state=c.core.empty_state(n);source=np.full(n,'',dtype='U240')
    tasks={(t['part'],t['start']):t for t in plan['tasks'] if t['branch']=='actual_on'}
    groups={}
    for i,(part,row) in enumerate(zip(selected['part'],selected['cache_row'])):
        groups.setdefault((int(part),int(row)//4096*4096),[]).append(i)
    for key,indices in groups.items():
        task=tasks[key];final,checkpoint=c.core.paths(task)
        path=final if final.exists() else checkpoint if checkpoint.exists() else None
        if path is None:continue
        data=c.core.read_task(task,plan);metadata=c.core.identity(task,data,plan,audit)
        old=c.core.read_state(path,metadata,len(data['raw']),require_complete=path==final,data=data)
        idx=np.asarray(indices);local=selected['cache_row'][idx]-task['start']
        np.testing.assert_array_equal(selected['particle_id'][idx],data['particle_id'][local])
        np.testing.assert_array_equal(selected['row'][idx],data['row'][local])
        np.testing.assert_array_equal(selected['actual'][idx],data['raw'][local])
        keep=old['status'][local]!=0;idx=idx[keep];local=local[keep]
        for name in state:state[name][idx]=old[name][local]
        source[idx]=str(path)
    return state,source


def main():
    assert int(os.environ.get('PBS_NCPUS','0'))>0
    sha=c.verify();target=c.ROOT/'sample.h5';assert not target.exists()
    # Revalidate the frozen proxy/model/map lineage on the compute node.
    old=json.loads((c.PILOT/'manifest.json').read_text())
    for path,expected in old['hashes'].items():assert c.core.digest(path)==expected,path
    pilot_sha=c.core.digest(c.PILOT/'manifest.json')
    plan=c.core.load_plan();audit=json.loads((c.FULL/'input_audit.json').read_text())
    ratios=plan['uniform_ratios'];paths=[Path(p['path']) for p in plan['parts']]
    predicted=np.zeros((9,6,3));outside=np.zeros((6,4));count=0
    for path in paths:
        with h5py.File(path) as f:
            assert f.attrs['manifest_sha256']==pilot_sha and f.attrs['complete']
            predicted+=f['predicted_particle_totals'][:];outside+=f['outside_coverage'][:]
            count+=len(f['mass'])
    assert count==plan['active_count'] and np.all(predicted[0,:,1:]>0)
    with h5py.File(c.PILOT/'pilot_sample.h5') as f:
        assert f.attrs['manifest_sha256']==pilot_sha
        np.testing.assert_allclose(predicted,f['predicted_particle_totals'][:],rtol=1e-12)
        exact_beam=f['predicted_beam_totals'][:]
    population=np.zeros(270,dtype='i8')
    for part,path in enumerate(paths):
        for data in chunks(path):population+=np.bincount(c.design_strata(data,ratios,predicted[0]),minlength=270)
        print('counted cache part',part,flush=True)
    assert population.sum()==count
    sample_n=c.allocation(population)
    rng=np.random.default_rng(c.SEED)
    ranks={h:np.sort(rng.choice(int(population[h]),int(sample_n[h]),replace=False)) for h in np.flatnonzero(population)}
    offset=np.zeros(270,dtype='i8');rows=[]
    for part,path in enumerate(paths):
        for data in chunks(path):
            strata=c.design_strata(data,ratios,predicted[0])
            for h in np.unique(strata):
                idx=np.flatnonzero(strata==h);lo,hi=offset[h],offset[h]+len(idx)
                pick=ranks[int(h)];a,b=np.searchsorted(pick,[lo,hi]);chosen=idx[pick[a:b]-lo]
                for j in chosen:rows.append({k:v[j] for k,v in data.items()}|dict(part=part,stratum=int(h)))
                offset[h]=hi
        print('sampled cache part',part,flush=True)
    np.testing.assert_array_equal(offset,population);assert len(rows)==c.DRAW_COUNT
    selected={k:np.asarray([row[k] for row in rows]) for k in rows[0]}
    order=np.lexsort((selected['cache_row'],selected['part']))
    selected={k:v[order] for k,v in selected.items()}
    assert len(np.unique(np.column_stack((selected['part'],selected['cache_row'])),axis=0))==c.DRAW_COUNT
    np.testing.assert_array_equal(np.bincount(selected['stratum'],minlength=270),sample_n)
    selected['inclusion_probability']=sample_n[selected['stratum']]/population[selected['stratum']]
    selected['uniform']=c.uniform(selected['actual'],ratios)
    selected['spatial_group_split']=c.core.split_group(selected['xyz'])
    # Labels are consulted only after the independent sample is fixed.
    reused,source=reuse_actual(selected,plan,audit)
    tmp=Path(str(target)+'.tmp')
    with h5py.File(tmp,'w') as f:
        f.attrs.update(complete=False,manifest_sha256=sha,seed=c.SEED,
            models_json=json.dumps(c.MODELS),branches_json=json.dumps(c.BRANCHES),regions_json=json.dumps(c.REGIONS),
            design='stratified simple random sampling without replacement',validation_only=True)
        for k,v in selected.items():f.create_dataset('sample/'+k,data=v,compression='gzip')
        for k,v in reused.items():f.create_dataset('reuse_actual/'+k,data=v,compression='gzip')
        f.create_dataset('reuse_actual/source',data=source.astype('S240'),compression='gzip')
        f.create_dataset('stratum_population',data=population);f.create_dataset('stratum_sample',data=sample_n)
        f.create_dataset('predicted_particle_totals',data=predicted);f.create_dataset('predicted_beam_totals',data=exact_beam)
        f.create_dataset('outside_coverage',data=outside)
        f.attrs.update(complete=True,active_count=count,active_mass_Msun=plan['active_mass_Msun'])
    os.replace(tmp,target)
    from geometry import prepare_weights,check_forward
    weights=prepare_weights();geometry_check=check_forward(selected,weights)
    c.save(c.ROOT/'sample_ready.json',dict(complete=True,manifest_sha256=sha,
        sample_sha256=c.core.digest(target),geometry_weights_sha256=c.core.digest(c.ROOT/'adjoint_weights.npy'),
        distinct_particles=c.DRAW_COUNT,physical_states=5*c.DRAW_COUNT,
        reused_actual_states=int(np.count_nonzero(reused['status'])),
        active_count=count,stratum_population=population,stratum_sample=sample_n,
        geometry_check=geometry_check,old_proxy_lineage_verified=True,validation_only=True))
    print('Prepared',c.DRAW_COUNT,'distinct particles; existing labels reused:',np.count_nonzero(reused['status']),flush=True)


if __name__=='__main__':main()

"""Outcome-blind nested draws and exact reuse, using cached reference arrays."""
import json
import os

import h5py
import numpy as np

import shield_common as c


def geometry(folder,n,key,shape):
    out=np.full((n,*shape),np.nan);seen=np.zeros(n,bool)
    for part in range(4):
        with np.load(folder/f'geometry/part{part:02d}.npz') as f:
            ids=f['ids'];assert not seen[ids].any();out[ids]=f[key];seen[ids]=True
    assert seen.all() and np.all(np.isfinite(out))
    return out


def main():
    assert int(os.environ.get('PBS_NCPUS','0'))>0
    digest=c.verify();assert not (c.ROOT/'prepared.json').exists()
    rng=np.random.default_rng(c.SEED);cohorts=[]
    for name,parent,count,targets in [('global',c.SOURCE,8192,None),
            ('outer',c.OUTER,4096,[64]*8+[448]*8)]:
        with h5py.File(parent/'sample.h5') as f:
            full={key:f['sample/'+key][:] for key in ('actual','slug','uniform','mass','nuclei','stratum','particle_id','lum_per_H')}
            population=f['stratum_population'][:]
        ids,allocation=c.nested(full['stratum'],population,count,rng,targets)
        data={key:value[ids] for key,value in full.items()}
        data['parent_row']=ids;data['pi']=allocation[data['stratum']]/population[data['stratum']]
        if name=='global':
            with np.load(c.ISO/'x8_failure_diagnostics_20261102/diagnostic_candidates.npz') as f:
                values=f['values'][ids];qualified=f['qualified'][ids]
            kernel=geometry(c.CLAIMS,len(full['mass']),'kernel',(2,19))[ids]
            sfr=geometry(c.SFR,len(full['mass']),'kernel',(80,))[ids]
        else:
            values=np.full((len(full['mass']),5,27),np.nan);qualified=np.zeros((len(values),5),bool)
            seen=np.zeros(values.shape[:2],bool)
            for part in range(8):
                with np.load(parent/f'direct/part{part:02d}.npz') as f:
                    task=f['task_ids'];i,b=task//5,task%5
                    assert not seen[i,b].any() and bool(f['complete'])
                    np.testing.assert_array_equal(f['particle_id'],full['particle_id'][i])
                    values[i,b]=f['values'];qualified[i,b]=f['qualified'];seen[i,b]=True
            assert seen.all();values,qualified=values[ids],qualified[ids]
            kernel=geometry(parent,len(full['mass']),'kernel',(4,19))[ids,2:4]
            sfr=np.zeros((len(ids),80))
        data.update(baseline_values=values,baseline_qualified=qualified,kernel=kernel,sfr_kernel=sfr)
        cohorts.append((name,data,population,allocation))
    # Both cohorts remain separate probability designs, including duplicate
    # physical particles independently selected in their two parent designs.
    with h5py.File(c.ROOT/'sample.h5','x') as f:
        f.attrs['manifest_sha256']=digest;f.attrs['complete']=False
        offset=0
        for name,data,population,allocation in cohorts:
            g=f.create_group(name)
            for key,value in data.items():g.create_dataset(key,data=value,compression='gzip',compression_opts=1)
            g.create_dataset('population',data=population);g.create_dataset('allocation',data=allocation)
            g.attrs['offset']=offset;offset+=len(data['mass'])
        f.attrs['complete']=True
    c.save(c.ROOT/'prepared.json',dict(complete=True,manifest_sha256=digest,
        sample_sha256=c.sha(c.ROOT/'sample.h5'),particles=offset,logical_states=offset*20,
        cohorts={name:len(data['mass']) for name,data,_,_ in cohorts},seed=c.SEED))
    print('Prepared',offset,'particles, two independent nested cohorts.',flush=True)


if __name__=='__main__':main()

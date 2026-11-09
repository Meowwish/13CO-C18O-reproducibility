"""Exposure and frozen-state table sensitivity, not a chemical-response bound."""
import os

import h5py
import numpy as np

import claim_common as c
from design_statistics import weighted_quantiles
from despotic.chemistry.visser_shielding import VisserCOShielding


def main():
    assert int(os.environ.get('PBS_NCPUS','0'))>0
    sha,_=c.ready()
    with h5py.File(c.SOURCE/'sample.h5') as f:
        raw={k:f['sample/'+k][:] for k in ('actual','slug','uniform')}
        weight=f['sample/nuclei'][:]/f['sample/inclusion_probability'][:]
    with np.load(c.REFERENCES[1][1]) as f:values,qualified=f['values'],f['qualified']
    kernel=np.zeros((len(weight),9))
    for part in range(4):
        with np.load(c.SOURCE/f'geometry/part{part:02d}.npz') as f:kernel[f['ids']]=f['kernel_weights']
    grid=VisserCOShielding(c.ISO/'chemTable_v4_full_direct/data/visser_co_shielding')
    records=[]
    for branch,name in enumerate(c.base.BRANCHES):
        r=raw['slug' if name=='slug_on' else 'uniform' if name.startswith('uniform') else 'actual']
        q=qualified[:,branch]
        rv=r[q];v=values[q,branch]
        element=rv[:,3]/np.maximum(rv[:,4],1e-30)
        molecular=v[:,3]/np.maximum(v[:,4],1e-40)
        # GOW.NH is cloud.colDen/2, as in the actual dxdt shielding call.
        nco=rv[:,1]/2*np.maximum(v[:,3],0);nh2=rv[:,1]/2*np.maximum(v[:,2],0)
        def ratio(carbon):
            a=grid.factor('13C16O',nco,nh2,b_co=.3,tex=v[:,0],c12_c13=carbon)
            b=grid.factor('12C18O',nco,nh2,b_co=.3,tex=v[:,0],c12_c13=carbon)
            return np.log10(np.maximum(a,1e-300)/np.maximum(b,1e-300))
        shift=ratio(molecular)-ratio(element)
        bounds=ratio(35.)-ratio(69.)
        for region,index in [('full_field',0),('inner_1_8',1)]:
            for line,col in [('13CO',7),('C18O',8)]:
                w=weight[q]*kernel[q,index]*np.maximum(v[:,col],0)
                assert w.sum()>0
                records.append(dict(branch=name,region=region,line=line,
                    out_of_carbon_table_light_fraction=float(w[(element<35)|(element>69)].sum()/w.sum()),
                    molecular_lookup_differs_fraction=float(w[np.abs(np.clip(element,35,69)-np.clip(molecular,35,69))>1e-8].sum()/w.sum()),
                    frozen_molecular_minus_elemental_log_shielding_ratio_quantiles=weighted_quantiles(shift,w),
                    frozen_bound35_minus69_log_shielding_ratio_quantiles=weighted_quantiles(bounds,w),
                    molecular_lookup_shift_exceeds_0p05_dex_fraction=float(w[np.abs(shift)>.05].sum()/w.sum())))
    c.save(c.ROOT/'shielding_exposure.json',dict(complete=True,manifest_sha256=sha,records=records,
        columns='GOW cloud-centre columns: supplied NH/2 times chemical abundance',
        scope='Frozen-state table diagnostic. No coupled equilibrium response calculated; not an uncertainty bound. Oxygen table ratio remains 557.'))


if __name__=='__main__':main()

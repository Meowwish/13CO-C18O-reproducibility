"""Preregistered paired direct effects; cohorts and nonresponse stay explicit."""
import json
import os

import h5py
import numpy as np

import shield_common as c


def main():
    assert int(os.environ.get('PBS_NCPUS','0'))>0
    digest,ready=c.ready();n=ready['particles']
    values=np.full((n*20,27),np.nan);qualified=np.zeros(n*20,bool);seen=np.zeros(n*20,bool)
    origins=np.zeros(n*20,'u1');counts=np.zeros((n*20,3),'i8')
    hashes={}
    for part in range(c.PARTITIONS):
        path=c.ROOT/f'direct/part{part:02d}.npz';hashes[str(path)]=c.sha(path)
        with np.load(path) as f:
            assert bool(f['complete']) and str(f['manifest_sha256'])==digest
            ids=f['task_ids'];assert not seen[ids].any() and np.all(ids%c.PARTITIONS==part)
            q=(f['status']==1)&np.all(f['final_returns']==1,axis=1)
            np.testing.assert_array_equal(q,f['qualified'])
            values[ids]=f['values'];qualified[ids]=q;seen[ids]=True
            origins[ids]=f['origin'];counts[ids]=f['lookup_counts']
    assert seen.all();values=values.reshape(n,4,5,27);qualified=qualified.reshape(n,4,5)
    catalogue=json.loads((c.SFR/'catalogue.json').read_text())['bins']
    records=[];profiles=[];shapes=[];sfr_rows=[]
    with h5py.File(c.ROOT/'sample.h5') as f:
        for cohort in ('global','outer'):
            g=f[cohort];start=int(g.attrs['offset']);size=len(g['mass']);sl=slice(start,start+size)
            direct=values[sl,:,:,7:9];q=qualified[sl]
            strata,population=g['stratum'][:],g['population'][:]
            design=c.stat.groups(strata,population);weight=g['nuclei'][:]/g['pi'][:]
            kernel=g['kernel'][:];pred=g['lum_per_H'][:,[0,1,3,4,5],1:3]
            def stats(m,b,mask,operator=1):return c.ratio(direct[:,m,b],mask,kernel[:,operator],weight)
            def interval(point,score):return c.stat.interval(float(point),score,design)
            def missing(mask,b,operator):
                wk=weight[:,None]*kernel[:,operator]
                total=wk.T@np.maximum(pred[:,b],0.)
                lost=(wk*(~mask)[:,None]).T@np.maximum(pred[:,b],0.)
                return np.divide(lost,total,out=np.full_like(lost,np.nan),where=total>0)
            for mode,mode_name in enumerate(c.MODES):
                for b,branch in enumerate(c.BRANCHES):
                    operator=0 if b<2 else 1
                    point,score=stats(mode,b,q[:,mode,b],operator)
                    miss=missing(q[:,mode,b],b,operator)
                    for j,region in enumerate(c.REGIONS):
                        profiles.append(dict(cohort=cohort,mode=mode_name,branch=branch,region=region,
                            log_energy_ratio=interval(point[j],score[:,j]),missing_baseline_predicted_positive_light=miss[j]))
                    if mode:
                        mask=q[:,mode,b]&q[:,0,b]
                        one,so=stats(mode,b,mask,operator);zero,sz=stats(0,b,mask,operator)
                        for j,region in enumerate(c.REGIONS):
                            records.append(dict(cohort=cohort,mode=mode_name,effect='absolute_ratio_shift',branch=branch,region=region,
                                change_dex=interval(one[j]-zero[j],so[:,j]-sz[:,j])))
                for comparison,(one,two) in c.PAIRS.items():
                    operator=0 if comparison=='source' else 1
                    mask=q[:,mode,one]&q[:,mode,two]&q[:,0,one]&q[:,0,two]
                    a,sa=stats(mode,one,mask,operator);b,sb=stats(mode,two,mask,operator)
                    za,sza=stats(0,one,mask,operator);zb,szb=stats(0,two,mask,operator)
                    for j,region in enumerate(c.REGIONS):
                        records.append(dict(cohort=cohort,mode=mode_name,effect=comparison,region=region,
                            paired_response_dex=interval(a[j]-b[j],sa[:,j]-sb[:,j]),
                            response_change_dex=interval(a[j]-b[j]-za[j]+zb[j],sa[:,j]-sb[:,j]-sza[:,j]+szb[:,j]),
                            missing_baseline_predicted_positive_light=[missing(mask,one,operator)[j],missing(mask,two,operator)[j]]))
                for branch in (0,4):
                    mask=q[:,mode,branch]&q[:,0,branch]
                    operator=0 if branch==0 else 1
                    one,so=stats(mode,branch,mask,operator);zero,sz=stats(0,branch,mask,operator)
                    coefficients={}
                    for j in range(1,7):
                        a=np.zeros(19);a[j:j+2]=[-1,1];coefficients[f'annulus_{j+1}_{j+2}_minus_{j}_{j+1}']=a
                    a=np.zeros(19);a[8],a[14]=-1,1;coefficients['outer_endpoint_14_15_minus_8_9']=a
                    a=np.zeros(19);x=np.arange(8.5,15.);dx=x-x.mean();a[8:15]=dx/(dx@dx)
                    coefficients['outer_OLS_slope_dex_per_kpc']=a
                    for name,a in coefficients.items():
                        use=a!=0
                        shapes.append(dict(cohort=cohort,mode=mode_name,branch=c.BRANCHES[branch],contrast=name,
                            direct=interval(a[use]@one[use],so@a),
                            change=interval(a[use]@(one-zero)[use],(so-sz)@a)))
                if cohort=='global':
                    mask=np.all(q[:,mode,[0,2,3,4]],axis=1)&np.all(q[:,0,[0,2,3,4]],axis=1)
                    # Index the branch axis separately to preserve particle order.
                    assert mask.shape==(size,)
                    k=g['sfr_kernel'][:]
                    for branch in (0,2,3,4):
                        point,score=c.ratio(direct[:,mode,branch],mask,k,weight)
                        zero,sz=c.ratio(direct[:,0,branch],mask,k,weight)
                        for window in (1,10):
                            for radial in (0,'fixed_effect'):
                                a=c.stat.coefficients(catalogue,window,radial);use=a!=0
                                sfr_rows.append(dict(mode=mode_name,branch=c.BRANCHES[branch],window_Myr=window,radial=radial,
                                    direct=interval(a[use]@point[use],score@a),
                                    change=interval(a[use]@(point-zero)[use],(score-sz)@a)))
    c.save(c.ROOT/'review.json',dict(complete=True,manifest_sha256=digest,sample_sha256=ready['sample_sha256'],
        qualified_states=int(qualified.sum()),logical_states=int(qualified.size),reused_states=int(np.sum(origins==1)),
        profiles=profiles,paired_effects=records,shape_checks=shapes,sfr_slopes=sfr_rows,output_hashes=hashes,
        scientific_acceptance=False,scope='Conditional successful-state equilibrium-model carbon-lookup sensitivity; cohorts not pooled; oxygen tables remain fixed 557.'))
    (c.ROOT/'STATUS.md').write_text('Coupled shielding stress tests completed; review JSON before paper acceptance.\n'
        'Oxygen-table and off-table physical uncertainties are not closed by this experiment.\n')
    print('Paired shielding review saved; no automatic scientific acceptance.',flush=True)


if __name__=='__main__':main()

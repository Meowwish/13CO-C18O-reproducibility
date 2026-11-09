"""Independent reference audit and selection/quantile precision assessment."""
import json
import os

import h5py
import numpy as np
from scipy.stats import norm

import selection_common as c
from selection_design import total_and_covariance,selection,selected_bounds,difference_bounds,catalogue_bounds,sum_ratio_bounds
from selection_statistics import detection,summaries
from selection_bootstrap import replicate_weights


def interval(replicas):
    x=np.asarray(replicas)
    return np.column_stack((
        np.quantile(np.where(np.isnan(x[:,:,0]),-np.inf,x[:,:,0]),.0125,axis=0,method='lower'),
        np.quantile(np.where(np.isnan(x[:,:,1]),np.inf,x[:,:,1]),.9875,axis=0,method='higher')))


def chain_bounds(lo,hi,weights,mandatory,possible,proxy=False):
    lc,mc=(slice(2,4),slice(6,8)) if proxy else (slice(0,2),slice(4,6))
    line=catalogue_bounds(lo[:,lc],hi[:,lc],weights,1000,mandatory,possible)[2]
    molecule=sum_ratio_bounds(lo[:,mc],hi[:,mc],weights,mandatory,possible)
    effective=np.array([line[0]/molecule[1],line[1]/molecule[0]]) if molecule[0]>0 else np.array([np.nan,np.nan])
    return np.array([line,molecule,effective])


def compare_chains(a,b):
    with np.errstate(divide='ignore',invalid='ignore'):
        return np.column_stack((np.log10(a[:,0]/b[:,1]),np.log10(a[:,1]/b[:,0])))


def effective_n(weight,mask):
    w=weight[mask]
    return float(w.sum()**2/(w@w)) if len(w) else 0.


def main():
    assert int(os.environ.get('PBS_NCPUS','0'))>0
    manifest,ready=c.ready();assert not (c.ROOT/'review.json').exists()
    with h5py.File(c.ROOT/'sample.h5') as f:
        baseline,full_proxy=f['baseline'][:],f['full_proxy'][:]
        populations,allocations=f['population'][:],f['allocation'][:]
        records={k:f['records/'+k][:] for k in ('particle','aperture','stratum','pi','kernel','unknown')}
        values,qualified,status=[f['sample/'+k][:] for k in ('values','qualified','status')]
        prediction,mass,nuclei=[f['sample/'+k][:] for k in ('prediction','mass','nuclei')]
        tasks=f['tasks'][:];needed=f['sample/needed'][:];factors=f['conversion_line'][:]
        xy,strata,pixel_pi=[f['apertures/'+k][:] for k in ('xy','stratum','inclusion_probability')]
        pixel_population=f['apertures/population'][:];pixel_allocation=f['apertures/allocation'][:]
    A=len(xy);assert A==c.APERTURES
    seen=np.zeros(status.size,bool)
    for part in range(c.PARTITIONS):
        with np.load(c.ROOT/f'direct/part{part:02d}.npz') as f:
            assert bool(f['complete']) and json.loads(str(f['metadata']))['sample_sha256']==ready['sample_sha256']
            ids=f['task_ids'];np.testing.assert_array_equal(ids,tasks[part::c.PARTITIONS])
            assert not seen[ids].any() and np.all(status.ravel()[ids]==0) and needed.ravel()[ids].all()
            np.testing.assert_array_equal(f['qualified'],(f['status']==1)&np.all(f['final_returns']==1,axis=1))
            status.ravel()[ids],qualified.ravel()[ids],values.reshape(-1,6)[ids]=f['status'],f['qualified'],f['values'][:,3:9]
            seen[ids]=True
    assert seen.sum()==len(tasks) and np.all(status[needed]!=0)
    assert c.known_models(status,qualified).all()
    total=baseline.copy();covariance=np.zeros((A,48,48));maxdiff=0.;maxcv=0.
    p=records['particle'];z=np.zeros((len(p),6,8))
    for model in range(6):
        b=c.MODEL_BRANCH[model];q=qualified[:,b] if model<2 else qualified[:,[0,2,3,4]].all(axis=1)
        w=records['kernel'][:,int(model>=2)]*records['unknown'][:,model]
        pl=nuclei[p,None]*prediction[p,model,4:6]*factors;pm=mass[p,None]*prediction[p,model,1:3]
        dl=nuclei[p,None]*np.where(q[p,None],values[p,b,4:6],0.)*factors
        dm=mass[p,None]*np.where(q[p,None],values[p,b,1:3],0.)
        z[:,model,:2]=w[:,None]*(dl-pl);z[:,model,2:4]=-w[:,None]*(~q[p,None])*pl
        z[:,model,4:6]=w[:,None]*(dm-pm);z[:,model,6:8]=-w[:,None]*(~q[p,None])*pm
    for a in range(A):
        use=records['aperture']==a
        if not use.any():assert populations[a].sum()==0;continue
        sh,pi=records['stratum'][use],records['pi'][use]
        np.testing.assert_array_equal(np.bincount(sh,minlength=16),allocations[a])
        np.testing.assert_array_equal(pi,allocations[a,sh]/populations[a,sh])
        x=z[use].reshape(-1,48)
        t,v=total_and_covariance(baseline[a].ravel(),x,pi,sh,populations[a]);total[a]=t.reshape(6,8);covariance[a]=v
        independent=baseline[a].ravel().copy();cv=np.zeros((48,48))
        for h in np.unique(sh):
            row=x[sh==h];n=len(row);N=populations[a,h]
            independent+=N*row.mean(axis=0)
            if n<N:
                centred=row-row.mean(axis=0);cv+=N*N*(1-n/N)*(centred.T@centred)/(n*(n-1))
        np.testing.assert_allclose(t,independent,rtol=2e-11,atol=1e-20)
        np.testing.assert_allclose(v,cv,rtol=2e-10,atol=1e-24)
        maxdiff=max(maxdiff,float(np.max(abs(t-independent)/(abs(t)+1e-100))))
        maxcv=max(maxcv,float(np.max(abs(v-cv))))
    family=A*48;critical=float(norm.isf(.025/(2*family)))
    se=np.sqrt(np.maximum(np.diagonal(covariance,axis1=1,axis2=2),0)).reshape(A,6,8)
    low,high=total-critical*se,total+critical*se
    c.base.atomic_npz(c.ROOT/'aperture_results.npz',estimates=total,joint_covariance=covariance,reference_low=low,
        reference_high=high,full_proxy=full_proxy,xy=xy,pixel_weight=1/pixel_pi,reference_band_critical=np.asarray(critical))
    radius=np.hypot(*xy.T);weight=1/pixel_pi;rng=np.random.default_rng(c.SEED+1)
    statistics=('median_local_ratio','ratio_of_line_medians','ratio_of_line_sums','detection_fraction')
    rows=[];migrations=[];pixel_checks=[];chains=[]
    for model,name in enumerate(c.MODELS):
        for a in range(A):
            t=total[a,model];block=covariance[a,model*8:(model+1)*8,model*8:(model+1)*8]
            if np.all(t[:4]>0):
                g=np.zeros(8);g[:4]=np.array([1/t[0],-1/t[1],-1/t[2],1/t[3]])/np.log(10)
                point=np.log10(t[0]/t[1])-np.log10(t[2]/t[3]);sigma=float(np.sqrt(max(0.,g@block@g)))
                ci=[point-1.96*sigma,point+1.96*sigma]
            else:point=sigma=ci=None
            pixel_checks.append(dict(model=name,aperture=a,radius_kpc=radius[a],ratio_error_dex=point,
                reference_sampling_se=sigma,pointwise_reference_interval_95=ci))
    for radial in range(8,15):
        use=np.flatnonzero((radius>=radial)&(radius<radial+1));w=weight[use];s=strata[use]
        np.testing.assert_array_equal(np.bincount(s,minlength=42)[np.unique(s)],pixel_allocation[np.unique(s)])
        bootstrap=replicate_weights(s,pixel_population,rng,800)*w[None,:]
        for model,name in enumerate(c.MODELS):
            dl,pl=total[use,model,:2],total[use,model,2:4]
            dlo,dhi=low[use,model,:2],high[use,model,:2]
            plo,phi=low[use,model,2:4],high[use,model,2:4]
            for mode in ('dual','dual_and_ratio_snr'):
                for depth in c.DEPTHS:
                    md,mp=detection(dl,depth,mode),detection(pl,depth,mode)
                    dm,dp=selection(dlo,dhi,depth,mode);pm,pp=selection(plo,phi,depth,mode)
                    certain=(dm&(~pp))|(pm&(~dp));possible=(dp&(~pm))|(pp&(~dm))
                    migrations.append(dict(model=name,annulus_kpc=[radial,radial+1],selection=mode,depth=depth,
                        direct_detection_count_bounds=[int(dm.sum()),int(dp.sum())],proxy_detection_count_bounds=[int(pm.sum()),int(pp.sum())],
                        weighted_changed_fraction_reference_bounds=[float(w[certain].sum()/w.sum()),float(w[possible].sum()/w.sum())],
                        effective_certain_detection_counts=[effective_n(w,dm),effective_n(w,pm)]))
                    one,two=summaries(dl,w,md),summaries(pl,w,mp)
                    bounds=difference_bounds(selected_bounds(dlo,dhi,w,depth,mode),selected_bounds(plo,phi,w,depth,mode))
                    eligible=min(dm.sum(),pm.sum())>=10;ci=np.full((4,2),np.nan)
                    if eligible:
                        ci=interval([difference_bounds(selected_bounds(dlo,dhi,bw,depth,mode),selected_bounds(plo,phi,bw,depth,mode)) for bw in bootstrap])
                    for j,stat in enumerate(statistics):
                        point=np.log10(one[j]/two[j]) if j<3 and one[j]>0 and two[j]>0 else one[j]-two[j] if j==3 else np.nan
                        rows.append(dict(model=name,annulus_kpc=[radial,radial+1],selection=mode,depth=depth,statistic=stat,
                            direct_point=one[j],matched_surrogate_point=two[j],error_point=point,error_unit='dex' if j<3 else 'fraction',
                            reference_only_error_bounds=bounds[j],reference_and_pixel_sampling_interval_95=ci[j],
                            sufficient_certain_detections=bool(eligible),interval_within_0p05_dex=bool(eligible and j<3 and np.isfinite(ci[j]).all() and ci[j,0]>=-.05 and ci[j,1]<=.05)))
        for mode in ('own_D1000','common_controls_D1000'):
            models=range(6) if mode=='own_D1000' else range(2,6)
            for model in models:
                others=[model] if mode=='own_D1000' else list(range(2,6))
                dm=np.all([detection(low[use,m,:2],1000) for m in others],axis=0)
                dp=np.all([detection(high[use,m,:2],1000) for m in others],axis=0)
                pm=np.all([detection(low[use,m,2:4],1000) for m in others],axis=0)
                pp=np.all([detection(high[use,m,2:4],1000) for m in others],axis=0)
                eligible=min(dm.sum(),pm.sum())>=10
                def evaluate_chain(bw):
                    return compare_chains(chain_bounds(low[use,model],high[use,model],bw,dm,dp),
                        chain_bounds(low[use,model],high[use,model],bw,pm,pp,True))
                bounds=evaluate_chain(w);ci=interval([evaluate_chain(bw) for bw in bootstrap]) if eligible else np.full((3,2),np.nan)
                chains.append(dict(model=c.MODELS[model],annulus_kpc=[radial,radial+1],mode=mode,
                    observables=['line_ratio','molecule_ratio','emission_per_molecule_ratio'],reference_only_error_bounds_dex=bounds,
                    reference_and_pixel_sampling_interval_95=ci,direct_detection_count_bounds=[int(dm.sum()),int(dp.sum())],
                    proxy_detection_count_bounds=[int(pm.sum()),int(pp.sum())],sufficient_certain_detections=bool(eligible),
                    interval_within_0p05_dex=[bool(eligible and np.isfinite(x).all() and x[0]>=-.05 and x[1]<=.05) for x in ci]))
    c.save(c.ROOT/'review.json',dict(complete=True,manifest_sha256=manifest,sample_sha256=ready['sample_sha256'],
        estimated_apertures=A,new_states=len(tasks),total_selected_qualified_states=int(qualified.sum()),
        pixel_checks=pixel_checks,selection_migration=migrations,selected_statistics=rows,selected_chains=chains,
        reference_band_family=family,reference_band_critical=critical,pixel_bootstrap_replicates=800,
        pixel_bootstrap='Positive rescaled multinomial weights with finite-population correction; census strata fixed.',
        automatic_scientific_acceptance=False,model_training_performed=False,
        scope='Independent D1000-targeted full-footprint cohort, exact archived states plus probability-sampled residuals. '
              'Intervals combine approximate simultaneous reference bands and pointwise stratified pixel-bootstrap uncertainty. '
              'No pooling with the broad cohort, no complete pixel census, and no bound on unknown failed-state light.'))
    c.save(c.ROOT/'independent_audit.json',dict(passed=True,joint_total_components=A*48,
        maximum_relative_total_difference=maxdiff,maximum_absolute_covariance_difference=maxcv,
        review_sha256=c.sha(c.ROOT/'review.json'),scope='Independent stratum accumulation of totals and full cross-line/model covariance.'))
    print('Reviewed',A,'apertures;',len(rows),'selection-statistic records; no automatic acceptance.',flush=True)


if __name__=='__main__':main()

"""Population-weighted paired validation with explicit numerical nonresponse."""
from collections import Counter
import json
import os

import h5py
import numpy as np
import common as c
from design_statistics import joint_covariance,interval,log_metrics,ess,weighted_quantiles

PAIRS=dict(source=(1,0),source_vs_paired=(1,2),exchange=(2,3),
    uniform_exchange=(4,5),pattern_on=(2,4),pattern_off=(3,5))


def assess(models,direct,proxy,valid,weight,mass_weight,strata,population,sample_n,exact,freq):
    use=np.all(valid[:,models],axis=1)
    contributing=weight>0;complete=bool(np.all(use|~contributing));k=len(models)
    dy=np.where(use[:,None,None],direct[:,models],0.)*weight[:,None,None]
    pp=np.where(use[:,None,None],proxy[:,models],0.)*weight[:,None,None]
    rows=np.column_stack((dy.reshape(len(weight),-1),pp.reshape(len(weight),-1)))
    covariance=joint_covariance(rows,strata,population,sample_n)
    sd,sp=dy.sum(axis=0),pp.sum(axis=0)
    d=exact[models]+sd-sp if complete else sd
    p=exact[models] if complete else sp
    quantities=log_metrics(d,p,complete,freq)
    full_proxy=(proxy[:,models]*weight[:,None,None]).sum(axis=0)
    missing_proxy=(proxy[~use][:,models]*weight[~use,None,None]).sum(axis=0)
    denominator=mass_weight.sum()
    with np.errstate(invalid='ignore',divide='ignore'):
        missing=np.divide(missing_proxy,full_proxy,out=np.full_like(full_proxy,np.nan),where=full_proxy>0)
    result=dict(models=[c.MODELS[i] for i in models],contributing_particles=int(contributing.sum()),
        common_success_contributing_particles=int(np.sum(contributing&use)),
        all_contributing_sampled_references_available=complete,
        scope='proxy_plus_design_weighted_residual_population_estimate' if complete else 'matched_success_conditional_design_weighted_diagnostic',
        full_population_accuracy_claim=False,
        missing_design_mass_fraction=float(mass_weight[~use].sum()/denominator) if denominator>0 else None,
        common_missing_predicted_line_fractions=missing,
        direct_line_totals=d,proxy_comparison_totals=p,
        line_errors={},ratio_errors={},direct_ratio_log10={},effective_luminosity_samples={},largest_luminosity_weight_fraction={})
    for j,model in enumerate(models):
        name=c.MODELS[model]
        result['line_errors'][name]=[interval(quantities['line_error'][3*j+l],
            quantities['line_error_gradient'][3*j+l],covariance) for l in range(3)]
        point=quantities['direct_ratio'][j]-quantities['proxy_ratio'][j]
        grad=quantities['direct_ratio_gradient'][j]-quantities['proxy_ratio_gradient'][j]
        result['ratio_errors'][name]=interval(point,grad,covariance)
        result['direct_ratio_log10'][name]=interval(quantities['direct_ratio'][j],quantities['direct_ratio_gradient'][j],covariance)
        result['effective_luminosity_samples'][name]=[ess(np.maximum(dy[:,j,l],0)) for l in range(3)]
        result['largest_luminosity_weight_fraction'][name]=[
            float(np.max(np.maximum(dy[:,j,l],0))/np.maximum(dy[:,j,l],0).sum()) if np.maximum(dy[:,j,l],0).sum()>0 else None for l in range(3)]
    if k==2:
        effect=quantities['direct_ratio'][0]-quantities['direct_ratio'][1]
        effect_grad=quantities['direct_ratio_gradient'][0]-quantities['direct_ratio_gradient'][1]
        pred_effect=quantities['proxy_ratio'][0]-quantities['proxy_ratio'][1]
        pred_grad=quantities['proxy_ratio_gradient'][0]-quantities['proxy_ratio_gradient'][1]
        result['direct_response_log10']=interval(effect,effect_grad,covariance)
        result['proxy_response_log10']=float(pred_effect)
        result['response_error_dex']=interval(effect-pred_effect,effect_grad-pred_grad,covariance)
    result['review_scale_dex']=.02 if k==2 else .05
    error=result.get('response_error_dex',next(iter(result['ratio_errors'].values())))
    ci=error['interval_95'];scale=result['review_scale_dex']
    result['interval_within_review_scale']=bool(ci is not None and ci[0]>=-scale and ci[1]<=scale)
    result['population_acceptance']='withheld_numerical_nonresponse' if not complete else 'requires_scientific_and_precision_review'
    return result


def main():
    assert int(os.environ.get('PBS_NCPUS','0'))>0
    sha,ready=c.ready();target=c.ROOT/'review.json';assert not target.exists()
    with h5py.File(c.ROOT/'sample.h5') as f:
        data={k:f['sample/'+k][:] for k in ('mass','nuclei','xyz','inclusion_probability','stratum','particle_id')}
        population=f['stratum_population'][:];sample_n=f['stratum_sample'][:]
        proxy=f['sample/lum_per_H'][:]
        exact={k:f['predicted_'+k+'_totals'][:] for k in ('particle','beam')}
    n=len(data['mass']);state={key:value.reshape((n,5,*value.shape[1:])) for key,value in c.core.empty_state(n*5).items()}
    seen=np.zeros(n*5,bool);origin=np.zeros((n,5),'u1');errors=Counter()
    for part in range(8):
        with np.load(c.ROOT/f'direct/part{part:02d}.npz',allow_pickle=False) as f:
            meta=json.loads(str(f['metadata']));assert meta['manifest_sha256']==sha and meta['sample_sha256']==ready['sample_sha256']
            assert bool(f['complete']);ids=f['task_ids'];assert np.all(ids%8==part) and not seen[ids].any()
            np.testing.assert_array_equal(f['particle_id'],data['particle_id'][ids//5])
            for key in state:state[key].reshape((n*5,*state[key].shape[2:]))[ids]=f[key]
            origin.reshape(-1)[ids]=f['origin'];seen[ids]=True
    assert seen.all() and np.all(state['status']!=0)
    assert np.all(np.isin(origin,[1,2]))
    assert np.all(np.isnan(state['values'][state['status']<0]))
    assert np.all(np.isfinite(state['values'][state['status']==1]))
    kernel=np.full((n,9),np.nan);seen_geometry=np.zeros(n,bool)
    for part in range(4):
        with np.load(c.ROOT/f'geometry/part{part:02d}.npz',allow_pickle=False) as f:
            assert str(f['manifest_sha256'])==sha and str(f['sample_sha256'])==ready['sample_sha256'] and bool(f['complete'])
            ids=f['ids'];assert np.all(ids%4==part) and not seen_geometry[ids].any()
            kernel[ids]=f['kernel_weights'];seen_geometry[ids]=True
    assert seen_geometry.all() and np.all(np.isfinite(kernel))
    frequencies=json.loads((c.ISO/'highres_snapshot575_temporal/config.json').read_text())['projection']['line_frequencies_Hz']
    freq=3*np.log10(frequencies['C18O']/frequencies['13CO'])
    valid=(state['status']==1)&np.all(np.isfinite(state['values']),axis=2)
    direct=state['values'][:,c.MODEL_BRANCH,6:9];v=valid[:,c.MODEL_BRANCH]
    inv=1/data['inclusion_probability'];particle=c.region_masks(data['xyz']).astype(float)
    result=dict(complete=True,manifest_sha256=sha,sample_sha256=ready['sample_sha256'],
        distinct_particles=n,physical_states=5*n,
        interval_method='Stratified SRSWOR joint total covariance with finite-population correction; first-order delta-method nominal 95% intervals',
        manuscript_acceptance=False,model_training_performed=False,branches={},results=[],strict_common_completeness=[],opacity_diagnostics=[])
    for b,name in enumerate(c.BRANCHES):
        status=state['status'][:,b];err=state['error'][:,b]
        errors.update(x.decode(errors='replace') for x in err[status<0])
        result['branches'][name]=dict(success=int(np.sum(status==1)),failed=int(np.sum(status==-1)),timeout=int(np.sum(status==-2)),
            reused=int(np.sum(origin[:,b]==1)),newly_attempted=int(np.sum(origin[:,b]==2)),
            quality_flagged_success=int(np.sum((status==1)&(state['quality_flags'][:,b]!=0))))
    for geometry,factors in (('particle',particle),('beam',kernel)):
        for r,region in enumerate(c.REGIONS):
            factor=factors[:,r];weight=inv*data['nuclei']*factor;mass_weight=inv*data['mass']*factor
            common=np.all(valid,axis=1);denom=(weight[:,None,None]*proxy).sum(axis=0)
            missed=(weight[~common,None,None]*proxy[~common]).sum(axis=0)
            with np.errstate(invalid='ignore',divide='ignore'):missed_fraction=missed/denom
            result['strict_common_completeness'].append(dict(geometry=geometry,region=region,
                all_five_success_contributing=int(np.sum(common&(factor>0))),
                common_missing_predicted_line_fractions=missed_fraction))
            for model in range(6):
                report=assess([model],direct,proxy,v,weight,mass_weight,data['stratum'],population,sample_n,exact[geometry][r],freq)
                result['results'].append(dict(geometry=geometry,region=region,comparison=c.MODELS[model],**report))
            for name,models in PAIRS.items():
                report=assess(list(models),direct,proxy,v,weight,mass_weight,data['stratum'],population,sample_n,exact[geometry][r],freq)
                result['results'].append(dict(geometry=geometry,region=region,comparison=name,**report))
            for b,branch in enumerate(c.BRANCHES):
                good=valid[:,b]
                for line,name in enumerate(('CO','13CO','C18O')):
                    weights=weight[good]*np.maximum(state['values'][good,b,6+line],0)
                    if weights.sum()<=0:continue
                    tau=state['values'][good,b,9+line];tex=state['values'][good,b,12+line];beta=state['values'][good,b,18+line]
                    result['opacity_diagnostics'].append(dict(geometry=geometry,region=region,branch=branch,line=name,
                        scope='conditional_success_positive_emission_weighted',ESS=ess(weights),
                        tau_gt_1_fraction=float(weights[tau>1].sum()/weights.sum()),
                        negative_tau_fraction=float(weights[tau<0].sum()/weights.sum()),
                        nonpositive_Tex_fraction=float(weights[tex<=0].sum()/weights.sum()),
                        tau_quantiles=weighted_quantiles(tau,weights),Tex_quantiles=weighted_quantiles(tex,weights),
                        beta_quantiles=weighted_quantiles(beta,weights)))
            print('reviewed',geometry,region,flush=True)
    result['error_counts']=dict(errors)
    result['notes']=[
        'The complete independent validation sample was fixed before consulting direct labels; proxies were frozen before this campaign.',
        'SRSWOR weights differ from the old with-replacement pilot; finite-population variance correction is explicit.',
        'Intervals quantify approximate sampling uncertainty, not physical model uncertainty or a simultaneous radial band.',
        'Predicted missing light is not an upper bound on missing true direct light; nonresponse is never imputed.',
        'A narrow interval on a matched successful mask does not certify full-population surrogate accuracy.',
        'No pixel-level, median, detection-selection or temporal-evolution accuracy claim is made.']
    c.save(target,result)
    lines=['# Independent x8/575 paired validation completed','',f'{n:,} distinct probability-sampled particles; {5*n:,} physical-state labels.','',
        '| Branch | Success | Failed | Timeout | Reused |','|---|---:|---:|---:|---:|']
    for b,d in result['branches'].items():lines.append(f"| {b} | {d['success']} | {d['failed']} | {d['timeout']} | {d['reused']} |")
    lines+=['','## Beam-matched whole-field and inner-disc response errors','',
        '| Region | Comparison | Error dex | Approximate 95% sampling interval | Scope |',
        '|---|---|---:|---|---|']
    for report in result['results']:
        if report['geometry']!='beam' or report['region'] not in ('full_field','inner_1_8') or report['comparison'] not in ('production_full','source','exchange','pattern_on','pattern_off'):continue
        error=report.get('response_error_dex',next(iter(report['ratio_errors'].values())))
        lines.append(f"| {report['region']} | {report['comparison']} | {error['point']} | {error['interval_95']} | {report['scope']} |")
    lines+=['','Detailed paired masks, finite-population sampling intervals, ESS, numerical failure/light accounting and tau/beta/Tex diagnostics are in `review.json`.',
        'No manuscript claim is automatically accepted. Review numerical nonresponse and radial precision before fitting or expanding.','']
    (c.ROOT/'RESULTS.md').write_text('\n'.join(lines))


if __name__=='__main__':main()

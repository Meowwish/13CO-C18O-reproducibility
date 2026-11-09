"""Signed paired factorial/sensitivity contrasts with finite-population covariance."""
import os
import h5py
import numpy as np
import extension as c


def contrast(values, qualified, indices, coefficient, ids, pi, data, population, geometry, label):
    common = np.all(qualified[indices], axis=1)
    lines = values[indices][:, :, 7:9]
    records = []
    for r, name in enumerate(c.REGIONS):
        weight = data['nuclei'][ids]*geometry[ids, r]/pi
        points, scores, totals = [], [], []
        for j in range(indices.shape[1]):
            p, s, t = c.log_ratio_and_score(lines[:,j], weight, common)
            points.append(p); scores.append(s); totals.append(t)
        valid = all(p is not None for p in points)
        result = c.interval(float(np.dot(coefficient, points)) if valid else None,
                            np.asarray(scores).T@coefficient if valid else None,
                            data['stratum'][ids], population)
        # Known fiducial surrogate light on failures; never an unknown-light bound.
        missing = []
        for model in (0, 1, 2, 3):
            light = weight[:, None]*data['lum_per_H'][ids, model, 1:3]
            missing.append(np.divide(light[~common].sum(axis=0), light.sum(axis=0),
                                     out=np.full(2,np.nan), where=light.sum(axis=0)>0))
        mass = data['mass'][ids]*geometry[ids, r]/pi
        records.append(dict(label=label, region=name, delta_log_R=result, qualified_particles=int(common.sum()),
                            sampled_particles=len(ids), signed_totals=totals,
                            known_fiducial_predicted_missing_light_fraction=missing,
                            missing_weighted_mass_fraction=float(mass[~common].sum()/mass.sum()) if mass.sum()>0 else None))
    return records


def main():
    assert int(os.environ.get('PBS_NCPUS','0')) > 0
    sha = c.verify(); data, population, geometry = c.sample_data()
    with h5py.File(c.ROOT/'cases.h5') as f:
        sample, cell, scenario, pi = (f[k][:] for k in ('sample_id','cell','scenario','inclusion_probability'))
        factorial, physical = f['factorial_ids'][:], f['physical_ids'][:]
        phys_alloc = f['physical_allocation'][:]
    values = np.full((len(sample),27),np.nan); qualified = np.zeros(len(sample),bool)
    status = np.zeros(len(sample),'i1'); seen = np.zeros(len(sample),bool)
    for part in range(8):
        with np.load(c.ROOT/f'direct/part{part:02d}.npz') as f:
            assert bool(f['complete']) and str(f['manifest_sha256']) == sha
            ids = f['case_ids']; assert not seen[ids].any(); seen[ids] = True
            values[ids] = f['values']; status[ids] = f['status']
            qualified[ids] = (f['status']==1)&np.all(f['final_returns']==1,axis=1)
    assert seen.all() and np.all(status != 0)
    lookup = {(int(sc),int(ce),int(s)): i for i,(sc,ce,s) in enumerate(zip(scenario,cell,sample))}
    assert len(lookup) == len(sample)
    indices = np.array([[lookup[0,ce,int(s)] for ce in range(4)] for s in factorial])
    fpi = pi[indices[:,0]]
    records = []
    contrasts = dict(source=[-1,1,0,0], isotope_at_Full_budget=[-1,0,1,0],
                     isotope_at_SLUG_budget=[0,1,0,-1], budget_at_Full_pattern=[-1,0,0,1],
                     budget_at_SLUG_pattern=[0,1,-1,0], interaction=[1,1,-1,-1])
    for label, coefficient in contrasts.items():
        records += contrast(values, qualified, indices, coefficient, factorial, fpi,
                            data, population, geometry, label)
    ppi = phys_alloc[data['stratum'][physical]]/population[data['stratum'][physical]]
    for sc, (label,*_) in enumerate(c.SCENARIOS[1:],1):
        for effect, cells, coefficient in (('source',(0,1),[-1,1,1,-1]),
                                           ('exchange',(0,4),[1,-1,-1,1])):
            # [perturbed first, perturbed second, fiducial first, fiducial second].
            ix = np.array([[lookup[sc,cells[0],int(s)],lookup[sc,cells[1],int(s)],
                            lookup[0,cells[0],int(s)],lookup[0,cells[1],int(s)]] for s in physical])
            records += contrast(values, qualified, ix, coefficient, physical, ppi,
                                data, population, geometry, label+' change_in_'+effect)
            direction = [-1,1,0,0] if effect=='source' else [1,-1,0,0]
            records += contrast(values, qualified, ix, direction, physical, ppi,
                                data, population, geometry, label+' '+effect)
    # Baseline paired effects on exactly the physical subset.
    ix = np.array([[lookup[0,ce,int(s)] for ce in (0,1,4)] for s in physical])
    for label, coeff in (('fiducial_source',[-1,1,0]),('fiducial_exchange',[1,0,-1])):
        records += contrast(values,qualified,ix,coeff,physical,ppi,data,population,geometry,label)
    c.save(c.ROOT/'control_review.json', dict(manifest_sha256=sha, total_cases=len(sample),
           qualified=int(qualified.sum()), failed_or_unqualified=int((~qualified).sum()), records=records,
           scope='Conditional retained common-state estimates, HT totals and linearized stratified sampling intervals; not numerical or physical-model error bounds.',
           oxygen_control='O factor scales 16O and 18O together; no recalibration after perturbation.',
           factorial='Both paths and interaction retained; no unique additive causal percentage assigned.'))
    (c.ROOT/'CONTROL_REVIEW.md').write_text(f'# Controlled physical comparisons\n\n{qualified.sum():,}/{len(sample):,} states qualified. '
        'See control_review.json for nine-region paired contrasts, both factorial paths, interaction, '
        'sampling intervals, and known fiducial missing-light diagnostics. Unresolved counterfactual emission remains unknown.\n')
    print('Control review complete', qualified.sum(), '/', len(sample))


if __name__ == '__main__': main()

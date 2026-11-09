"""Small no-solver checks of scientific transformations and covariance arithmetic."""
import ast
import json
from pathlib import Path
import subprocess
import h5py
import numpy as np
import extension as c


def main():
    for path in c.ROOT.glob('*.py'): ast.parse(path.read_text(),filename=str(path))
    subprocess.run(['bash','-n',str(c.ROOT/'run.pbs')],check=True)
    data,population,geometry=c.sample_data()
    for budget,pattern in ((data['actual'],data['slug']),(data['slug'],data['actual'])):
        result=c.cross(budget,pattern)
        np.testing.assert_array_equal(result[:,:3],budget[:,:3])
        for a,b in ((3,4),(5,6)):
            np.testing.assert_allclose(result[:,a]+result[:,b],budget[:,a]+budget[:,b],rtol=1e-14)
            np.testing.assert_allclose(result[:,b]/(result[:,a]+result[:,b]),pattern[:,b]/(pattern[:,a]+pattern[:,b]),rtol=1e-14)
    # Explicit zero rare-isotope end points must not create 0/0 isotope ratios.
    test=data['actual'][:3].copy();test[:,[4,6]]=0
    np.testing.assert_array_equal(c.cross(test,test),test)
    from itertools import combinations
    raw_score=np.array([1.,2.,5.,8.]);N=len(raw_score);n=2
    totals=[];estimates=[]
    for draw in combinations(range(N),n):
        contribution=raw_score[list(draw)]/(n/N)
        totals.append(contribution.sum())
        result=c.interval(0.,contribution,np.zeros(n,int),np.array([N]))
        estimates.append(result['standard_error']**2)
    np.testing.assert_allclose(np.mean(totals),raw_score.sum())
    np.testing.assert_allclose(np.mean(estimates),np.var(totals))
    point,score,total=c.log_ratio_and_score(np.array([[2.,1.],[np.nan,np.nan],[4.,2.]]),np.ones(3),np.array([1,0,1],bool))
    np.testing.assert_allclose(point,np.log10(2));np.testing.assert_array_equal(score,np.zeros(3))
    np.testing.assert_array_equal(total,[6.,3.])
    with h5py.File(c.ROOT/'cases.h5') as f:
        sample,cell,scenario,pi=(f[k][:] for k in ('sample_id','cell','scenario','inclusion_probability'))
        raw=f['raw'][:];factorial=f['factorial_ids'][:];physical=f['physical_ids'][:]
        assert len(sample)==110592 and len(factorial)==8192 and len(physical)==4096
        assert set(physical)<=set(factorial)
        for sc,(_,uv,cr,oxygen) in enumerate(c.SCENARIOS):
            for ce in ((0,1,2,3,4) if sc==0 else (0,1,4)):
                use=(scenario==sc)&(cell==ce);ids=sample[use]
                expected_ids=physical if sc>0 or ce==4 else factorial
                np.testing.assert_array_equal(ids,expected_ids)
                counts=np.bincount(data['stratum'][ids],minlength=len(population))
                assert np.all((population==0)|(counts>=2)|(counts==population))
                np.testing.assert_allclose(pi[use],counts[data['stratum'][ids]]/population[data['stratum'][ids]])
                expected=(data['slug'] if ce==1 else data['actual'])[ids].copy()
                if ce==2:expected=c.cross(data['actual'][ids],data['slug'][ids])
                if ce==3:expected=c.cross(data['slug'][ids],data['actual'][ids])
                expected[:,5:7]*=oxygen
                np.testing.assert_array_equal(raw[use],expected)
    # A stub exercises configuration forwarding without invoking DESPOTIC.
    import solve_controls as runner
    runner.BASE_CLOUD={'chi_Draine':1.7,'cosmic_ray_ionization_rate_s-1':1e-16}
    class Direct: CLOUD={}
    class Worker: DIRECT=Direct()
    previous_worker=runner.runtime.phase.worker
    previous_solve=runner.runtime.solve
    observed=[]
    def stub(task):
        observed.append((task,runner.runtime.phase.worker.DIRECT.CLOUD.copy()))
        return task
    runner.runtime.phase.worker=Worker()
    runner.runtime.solve=stub
    try:
        for sc in (1,4,0):runner.solve((0,sc,0,(1.,)*7))
        assert [row[0][1] for row in observed]==[1,1,1]
        np.testing.assert_allclose([row[1]['chi_Draine'] for row in observed],[.51,1.7,1.7])
        np.testing.assert_allclose([row[1]['cosmic_ray_ionization_rate_s-1'] for row in observed],[1e-16,5e-16,1e-16],rtol=1e-14,atol=0)
    finally:
        runner.runtime.phase.worker=previous_worker;runner.runtime.solve=previous_solve
    import time_sequence as ts
    native=np.zeros((9,9));native[4,4]=1
    np.testing.assert_allclose(ts.smooth(np.ones((9,9))),1)
    assert ts.smooth(native)[4,4]>0
    # Known negative SFR slope with a radius-dependent intercept, no snapshot I/O.
    import plot_galic_selection as selection
    previous_lattice,previous_sample=selection.lattice,selection.sample
    x=np.tile(np.repeat(np.arange(-2.75,-.74,.5),12),2)
    radius=np.repeat([1.5,2.5],60)
    xy=np.column_stack((radius,np.zeros(len(radius))))
    beam=np.ones((6,2,len(x)));beam[:,0]=10**(.2*radius-.1*x)
    selection.lattice=lambda:xy
    selection.sample=lambda field, positions:field
    try:
        bins,slopes=ts.sfr_slopes(beam,np.stack((10**x,10**x)))
        np.testing.assert_allclose([s['slope'] for s in slopes],-.1,atol=1e-14)
    finally:
        selection.lattice=previous_lattice;selection.sample=previous_sample
    configs=[]
    for entry in json.loads((c.ROOT/'inventory.json').read_text()):
        snap=entry['snapshot'];settings=ts.prod.load_settings(c.ROOT/f'time/{snap}/config.json')
        assert [str(p) for p in ts.prod.snapshot_files(settings)]==[f['path'] for f in entry['files']]
        assert all(str(c.ROOT/f'time/{snap}') in value for value in settings['outputs'].values())
        configs.append(snap)
    np.testing.assert_array_equal(configs,np.arange(551,576))
    c.save(c.ROOT/'local_checks.json',dict(passed=True,chemical_solves=0,
           checks=['Python AST and PBS shell syntax','isotope fractions and elemental budgets including zero rare abundance',
                   'HT weighting and finite-population variance by exhaustive small population enumeration',
                   'NaN failure masking and signed ratio score','all 110592 raw controls and final sampling probabilities',
                   'noncensus per-stratum n>=2','cloud perturbation reset and extended-protocol forwarding using stub',
                   'finite beam normalization','whole and within-radius SFR slopes with known synthetic radial intercept',
                   '25 epoch input/output paths from header inventory']))
    print('Local scientific checks passed; no chemistry or snapshot streaming on login.')


if __name__=='__main__':main()

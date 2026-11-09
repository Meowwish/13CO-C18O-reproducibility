"""Small statistical checks only; no projection, inference or chemistry."""
import ast
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import subprocess

import numpy as np

import claim_common as c


def import_geometry_worker():
    # Same spawn mechanism and lazy production import as the real workers;
    # no particle projection or model inference is run on the login node.
    import claim_geometry
    import process_snapshot563_lookup_v11 as production
    import workflow_config
    assert callable(claim_geometry.one) and callable(production.project_surface)
    return str(workflow_config.__file__)


def main():
    with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
        path=pool.submit(import_geometry_worker).result(timeout=45)
    assert path==str(c.ISO/'chemTable_v4_full_direct/scripts/workflow_config.py'),path
    rng=np.random.default_rng(20261105)
    n=40
    values=rng.uniform(.1,2,(n,2));prediction=values*rng.uniform(.8,1.2,(n,2))
    kernel=rng.uniform(0,1,(n,3));weight=rng.uniform(1,3,n)
    mask=np.ones(n,bool);mask[3]=False
    values[3]=np.nan
    result=c.estimate(values,prediction,mask,kernel,weight)
    d=np.where(mask[:,None],values,0.);p=np.where(mask[:,None],prediction,0.)
    D=(weight[:,None]*kernel).T@d
    np.testing.assert_allclose(result['direct'],np.log10(D[:,0]/D[:,1]))
    # Finite-difference multiplicity derivatives independently check the
    # shared-particle score that drives all cross-annulus covariance.
    for i in (0,3,17):
        step=1e-5;w2=weight.copy();w2[i]*=1+step
        changed=c.estimate(values,prediction,mask,kernel,w2)
        np.testing.assert_allclose((changed['direct']-result['direct'])/step,
            result['direct_score'][i],rtol=2e-5,atol=2e-10)
    design=[(np.arange(20),100),(np.arange(20,40),120)]
    a=np.array([-1.,1.,0.])
    record,sd,sp=c.contrast(result,a,design)
    expected=sum(len(ids)*(1-len(ids)/N)*np.var((result['direct_score']@a)[ids],ddof=1)
                 for ids,N in design)
    np.testing.assert_allclose(record['direct']['standard_error']**2,expected,rtol=1e-13)
    assert np.all(result['direct_score'][~mask]==0)
    null=c.estimate(values,values,mask,kernel,weight)
    np.testing.assert_allclose(null['direct_score']-null['proxy_score'],0,atol=0)
    # Preserve the published SFR coefficient definition and eligible bins.
    bins=json.loads((c.SFR/'catalogue.json').read_text())['bins']
    for window in (1,10):
        for radial in (0,'fixed_effect'):
            coef=c.sfr.coefficients(bins,window,radial)
            assert coef is not None and abs(coef.sum())<1e-12
    # Recover known distinct slopes for 12 sectors and six models. Empty
    # coefficients and four selected bins exercise the previous axis bug.
    x=np.array([-2.75,-2.25,-1.75,-1.25])
    dx=x-x.mean();coefficient=np.zeros(7);coefficient[[0,2,4,6]]=dx/(dx@dx)
    expected=np.arange(72).reshape(12,6)/200-.2
    logarithm=.7+expected[:,None,:]*x[None,:,None]
    leave=np.ones((12,7,6,2))
    for column,index in enumerate((0,2,4,6)):leave[:,index,:,0]=10**logarithm[:,column,:]
    np.testing.assert_allclose(c.spatial_slope_replicas(coefficient,leave),expected,atol=3e-16,rtol=1e-13)
    from despotic.chemistry.visser_shielding import VisserCOShielding
    grid=VisserCOShielding(c.ISO/'chemTable_v4_full_direct/data/visser_co_shielding')
    for species in ('13C16O','12C18O'):
        v=grid.factor(species,np.array([1e14,1e16]),np.array([1e20,1e21]),c12_c13=np.array([69.,103.]))
        fixed=grid.factor(species,np.array([1e14,1e16]),np.array([1e20,1e21]),c12_c13=69.)
        np.testing.assert_allclose(v,fixed,atol=0,rtol=0)
    for path in c.ROOT.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
    subprocess.run(['bash','-n',str(c.ROOT/'run.pbs')],check=True)
    c.save(c.ROOT/'local_checks.json',dict(passed=True,finite_difference_scores=True,
        spawned_geometry_import=True,workflow_config_path=str(c.ISO/'chemTable_v4_full_direct/scripts/workflow_config.py'),
        covariance_contrast=True,numerical_nonresponse=True,sfr_coefficient_replay=True,
        shielding_clipping_replay=True,scope='Small deterministic checks; no science result.'))
    print('Local statistical and shielding-lookup checks passed.')


if __name__=='__main__':main()

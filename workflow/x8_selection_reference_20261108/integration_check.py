"""Synthetic prepare-to-review integration; all predictions/labels are fixtures."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import h5py
import numpy as np


def run():
    import selection_common as c
    import selection_prepare as prep
    import selection_review as review
    with tempfile.TemporaryDirectory(prefix='selection-reference-test-') as folder:
        root=Path(folder)
        c.ISO=root;c.ROOT=root/'study';c.OLD=root/'old';c.PILOT=root/'pilot';c.CATALOGUE=root/'catalogue'
        c.APERTURES=7;c.PARTITIONS=2;c.TOP=1;c.TAIL=16
        for p in (c.ROOT,c.OLD/'direct',c.PILOT/'direct',c.CATALOGUE,
                  root/'x8_full_direct_575_20261028',root/'x8_bounded_reference_20261101',
                  root/'x8_failure_diagnostics_20261102',root/'x8_outer_validation_20261105/direct'):
            p.mkdir(parents=True,exist_ok=True)
        os.environ['PBS_NCPUS']='1'  # Synthetic fixtures only; no real model/solver called.
        c.verify=lambda:'synthetic-manifest'
        n=12;actual=np.tile([100.,1e21,1.,10.,1.,12.,1.],(n,1));actual[:,0]+=np.arange(n)
        slug=actual.copy();slug[:,3:]*=.8;ratios=[2.,3.]
        uniform=c.base.uniform(actual,ratios);pid=np.arange(n)+1000
        def fake_predict(raws):
            outputs=[]
            for raw in (raws[0],raws[1],raws[0],raws[0],raws[2],raws[2]):
                scale=(raw[:,0]-90)*raw[:,3]/100
                outputs.append(scale[:,None]*np.array([.1,.05,.02,1.,.3,.1]))
            return np.stack(outputs,axis=1)
        c.parent.predict=fake_predict
        pred=fake_predict((actual,slug,uniform))
        physical=pred[:,[0,1,3,4,5]]*(1+.01*np.arange(n)[:,None,None])
        all_q=np.ones((n,5),bool);all_q[2,0]=False;all_q[9,3]=False
        xy=np.column_stack((np.arange(8,15)+.5,np.zeros(7)))
        coefficients=.01*(1+np.arange(n))[None,:]*(1+.1*np.arange(7))[:,None]
        def fake_patches(points):return np.zeros((7,2,1,1)),np.zeros((7,4),'i8'),0.
        def fake_kernel(xyz,h,particle,aperture,patches,bounds):
            ids=xyz[particle,0].astype('i8')
            a=.01*(1+ids)*(1+.1*aperture)
            return np.column_stack((a,a*1.01))
        prep.make_patches=fake_patches;prep.evaluate=fake_kernel
        cached=np.zeros((7,6,4))
        for m in range(6):
            k=coefficients*(1.01 if m>=2 else 1.)
            cached[:,m,:2]=k@pred[:,m,4:6];cached[:,m,2:]=k@pred[:,m,1:3]
        population=np.zeros(42,'i8');population[np.arange(7)*6]=1
        catalogue=c.CATALOGUE/'catalogue.npz'
        np.savez_compressed(catalogue,xy=xy,lattice_index=np.arange(7),stratum=np.arange(7)*6,
            inclusion_probability=np.ones(7),population=population,allocation=population,predicted=cached)
        c.save(c.CATALOGUE/'prepared.json',dict(complete=True,catalogue_sha256=c.sha(catalogue)))
        sys.modules['plot_galic_selection']=SimpleNamespace(sample=lambda field,points:field)
        maps=[]
        for m,name in enumerate(c.MODELS):
            path=root/f'map{m}.npz';maps.append(path)
            labels=('C13O_abundance','C18O_abundance') if m<2 else (name+'_X13CO',name+'_XC18O')
            if m<2:np.savez(path,**dict(zip(labels,cached[:,m,2:].T)))
        control={}
        for m,name in enumerate(c.MODELS[2:],2):
            control.update({name+'_X13CO':cached[:,m,2],name+'_XC18O':cached[:,m,3]})
        np.savez(maps[2],**control);c.parent.MAPS=maps[:3]
        source=root/'cache.h5'
        with h5py.File(source,'w') as f:
            f.attrs['complete']=True
            for name,value in dict(actual=actual,slug=slug,mass=np.ones(n),nuclei=np.ones(n),
                xyz=np.column_stack((np.arange(n),np.zeros(n),np.zeros(n))),h=np.full(n,20000.),
                row=np.arange(n),particle_id=pid,lum_per_H=pred[:,:,3:]).items():f[name]=value
        c.save(root/'x8_full_direct_575_20261028/registry.json',dict(parts=[dict(path=str(source))],uniform_ratios=ratios))
        with h5py.File(c.OLD/'census.h5','w') as f:
            for name,value in dict(part=np.zeros(6,'i8'),cache_row=np.arange(6),actual=actual[:6],slug=slug[:6],particle_id=pid[:6]).items():f['particles/'+name]=value
            f['prediction']=pred[:6];f['conversion_line']=np.ones(2)
        c.save(c.OLD/'prepared.json',dict(census_sha256=c.sha(c.OLD/'census.h5')))
        initial=np.zeros((6,5),'i1');initial[0]=1;initial[2,0]=-1;initial[2,1]=1;initial[3,:4]=1;initial[4]=1
        def packed(rows,branches):
            z=np.full((len(rows),27),np.nan);z[:,3:9]=physical[rows,branches];return z
        for part in range(32):
            ids=np.arange(part,30,32);rows,branches=ids//5,ids%5;s=initial.ravel()[ids];q=s==1
            np.savez_compressed(c.OLD/f'direct/part{part:02d}.npz',task_ids=ids,status=s,qualified=q,
                final_returns=np.where(q[:,None],np.ones((len(ids),3),'i1'),-np.ones((len(ids),3),'i1')),values=packed(rows,branches))
        with h5py.File(c.PILOT/'sample.h5','w') as f:
            f['sample/census_row']=np.array([3]);f['apertures/xy']=xy
        c.save(c.PILOT/'prepared.json',dict(sample_sha256=c.sha(c.PILOT/'sample.h5')))
        np.savez_compressed(c.PILOT/'aperture_results.npz',full_proxy=cached)
        for part in range(32):
            ids=np.array([4],'i8') if part==0 else np.empty(0,'i8');rows=np.full(len(ids),3,'i8');branches=np.full(len(ids),4,'i8')
            np.savez_compressed(c.PILOT/f'direct/part{part:02d}.npz',complete=True,task_ids=ids,particle_id=pid[rows],
                status=np.ones(len(ids),'i1'),qualified=np.ones(len(ids),bool),final_returns=np.ones((len(ids),3),'i1'),values=packed(rows,branches))
        def small_sample(path,rows):
            with h5py.File(path,'w') as f:
                for name,value in dict(part=np.zeros(len(rows),'i8'),cache_row=np.array(rows),actual=actual[rows],slug=slug[rows],uniform=uniform[rows],particle_id=pid[rows]).items():f['sample/'+name]=value
        small_sample(root/'x8_bounded_reference_20261101/sample.h5',[1,6])
        z=np.full((2,5,27),np.nan);z[:,:,3:9]=physical[[1,6]]
        np.savez_compressed(root/'x8_failure_diagnostics_20261102/diagnostic_candidates.npz',values=z,qualified=np.ones((2,5),bool))
        small_sample(root/'x8_outer_validation_20261105/sample.h5',[7])
        for part in range(8):
            ids=np.arange(part,5,8);rows=np.full(len(ids),7,'i8');branches=ids%5
            np.savez_compressed(root/f'x8_outer_validation_20261105/direct/part{part:02d}.npz',complete=True,task_ids=ids,
                values=packed(rows,branches),qualified=np.ones(len(ids),bool),status=np.ones(len(ids),'i1'),final_returns=np.ones((len(ids),3),'i1'))
        prep.main()
        prepared=json.loads((c.ROOT/'prepared.json').read_text())
        c.ready=lambda:('synthetic-manifest',prepared)
        with h5py.File(c.ROOT/'sample.h5') as f:
            tasks=f['tasks'][:];rows=f['sample/cache_row'][:]
        for part in range(2):
            ids=tasks[part::2];r=rows[ids//5];b=ids%5;q=all_q[r,b]
            (c.ROOT/'direct').mkdir(exist_ok=True)
            np.savez_compressed(c.ROOT/f'direct/part{part:02d}.npz',complete=True,
                metadata=np.asarray(json.dumps(dict(sample_sha256=prepared['sample_sha256']))),task_ids=ids,
                status=np.where(q,1,-1).astype('i1'),qualified=q,final_returns=np.repeat(np.where(q,1,-1)[:,None],3,axis=1),values=packed(r,b))
        review.main()
        with np.load(c.ROOT/'aperture_results.npz') as f:found=f['estimates']
        expected=np.zeros_like(found)
        for m in range(6):
            branch=c.MODEL_BRANCH[m];q=all_q[:,branch] if m<2 else all_q[:,[0,2,3,4]].all(axis=1)
            k=coefficients*(1.01 if m>=2 else 1.)
            expected[:,m,:2]=k@np.where(q[:,None],physical[:,branch,4:6],0.)
            expected[:,m,2:4]=k@np.where(q[:,None],pred[:,m,4:6],0.)
            expected[:,m,4:6]=k@np.where(q[:,None],physical[:,branch,1:3],0.)
            expected[:,m,6:8]=k@np.where(q[:,None],pred[:,m,1:3],0.)
        np.testing.assert_allclose(found,expected,rtol=2e-12,atol=1e-13)
        assert json.loads((c.ROOT/'independent_audit.json').read_text())['passed']
        return dict(passed=True,synthetic_apertures=7,exact_projected_components=int(found.size),
                    reused_states=prepared['selected_reused_states'],new_fixture_states=len(tasks),chemical_solves=0)

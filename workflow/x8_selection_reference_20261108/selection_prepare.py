"""Stream exact support/known contributions; sample only remaining residuals."""
import json
import os

import h5py
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix

import selection_common as c
from kernel import make_patches,evaluate
from selection_design import allocate


def read_rows(dataset,indices):
    indices=np.asarray(indices)
    assert np.all(np.diff(indices)>0)
    result=np.empty((len(indices),)+dataset.shape[1:],dtype=dataset.dtype)
    for start in range(0,len(dataset),100000):
        lo,hi=np.searchsorted(indices,[start,start+100000])
        if hi>lo:result[lo:hi]=dataset[start:start+100000][indices[lo:hi]-start]
    return result


def archive_pool():
    """Preserve terminal nonresponse; add subsequent labels only to unknown states."""
    previous=json.loads((c.OLD/'prepared.json').read_text())
    assert c.sha(c.OLD/'census.h5')==previous['census_sha256']
    previous=json.loads((c.PILOT/'prepared.json').read_text())
    assert c.sha(c.PILOT/'sample.h5')==previous['sample_sha256']
    registry=json.loads((c.ISO/'x8_full_direct_575_20261028/registry.json').read_text())
    with h5py.File(c.OLD/'census.h5') as f:
        keys=f['particles/part'][:].astype('i8')*(1<<32)+f['particles/cache_row'][:]
        raw={k:f['particles/'+k][:] for k in ('actual','slug','particle_id')}
        prediction=f['prediction'][:]
    n=len(keys);assert np.all(np.diff(keys)>0)
    status=np.zeros((n,5),'i1');qualified=np.zeros((n,5),bool);values=np.full((n,5,6),np.nan)
    for part in range(32):
        with np.load(c.OLD/f'direct/part{part:02d}.npz') as f:
            ids=f['task_ids'];s,q=f['status'],f['qualified']
            np.testing.assert_array_equal(q,(s==1)&np.all(f['final_returns']==1,axis=1))
            status.ravel()[ids]=s;qualified.ravel()[ids]=q
            values.reshape(-1,6)[ids]=f['values'][:,3:9]
    with h5py.File(c.PILOT/'sample.h5') as f:
        census_row=f['sample/census_row'][:]
        for part in range(32):
            with np.load(c.PILOT/f'direct/part{part:02d}.npz') as z:
                assert bool(z['complete'])
                ids=z['task_ids'];old_ids=census_row[ids//5]*5+ids%5
                assert np.all(status.ravel()[old_ids]==0)
                np.testing.assert_array_equal(z['particle_id'],raw['particle_id'][old_ids//5])
                s,q=z['status'],z['qualified']
                np.testing.assert_array_equal(q,(s==1)&np.all(z['final_returns']==1,axis=1))
                status.ravel()[old_ids]=s;qualified.ravel()[old_ids]=q
                values.reshape(-1,6)[old_ids]=z['values'][:,3:9]
    # Independent references can add labels for particles outside old apertures.
    extras=[]
    with h5py.File(c.ISO/'x8_bounded_reference_20261101/sample.h5') as f:
        small={k:f['sample/'+k][:] for k in ('part','cache_row','actual','slug','uniform','particle_id')}
    with np.load(c.ISO/'x8_failure_diagnostics_20261102/diagnostic_candidates.npz') as f:
        small.update(values=f['values'][:,:,3:9],qualified=f['qualified'])
    extras.append(small)
    with h5py.File(c.ISO/'x8_outer_validation_20261105/sample.h5') as f:
        small={k:f['sample/'+k][:] for k in ('part','cache_row','actual','slug','uniform','particle_id')}
    nv=len(small['part']);sv=np.full((nv*5,6),np.nan);sq=np.zeros(nv*5,bool)
    for part in range(8):
        with np.load(c.ISO/f'x8_outer_validation_20261105/direct/part{part:02d}.npz') as f:
            assert bool(f['complete'])
            np.testing.assert_array_equal(f['qualified'],(f['status']==1)&np.all(f['final_returns']==1,axis=1))
            ids=f['task_ids'];sv[ids]=f['values'][:,3:9];sq[ids]=f['qualified']
    small.update(values=sv.reshape(nv,5,6),qualified=sq.reshape(nv,5));extras.append(small)
    for small in extras:
        np.testing.assert_array_equal(small['uniform'],c.base.uniform(small['actual'],registry['uniform_ratios']))
        sk=small['part'].astype('i8')*(1<<32)+small['cache_row'];order=np.argsort(sk)
        sk=sk[order];small={k:v[order] for k,v in small.items()}
        loc=np.searchsorted(keys,sk);hit=loc<len(keys);hit[hit]&=keys[loc[hit]]==sk[hit]
        for name in ('actual','slug','particle_id'):
            np.testing.assert_array_equal(raw[name][loc[hit]],small[name][hit])
        if (~hit).any():
            use=~hit;fresh=sk[use]
            pk=c.parent.predict((small['actual'][use],small['slug'][use],small['uniform'][use]))
            joined=np.r_[keys,fresh];sort=np.argsort(joined)
            keys=joined[sort]
            prediction=np.concatenate((prediction,pk))[sort]
            for name in raw:raw[name]=np.concatenate((raw[name],small[name][use]))[sort]
            status=np.concatenate((status,np.zeros((use.sum(),5),'i1')))[sort]
            qualified=np.concatenate((qualified,np.zeros((use.sum(),5),bool)))[sort]
            values=np.concatenate((values,np.full((use.sum(),5,6),np.nan)))[sort]
        loc=np.searchsorted(keys,sk)
        for branch in range(5):
            use=small['qualified'][:,branch]&(status[loc,branch]==0)
            status[loc[use],branch]=1;qualified[loc[use],branch]=True
            values[loc[use],branch]=small['values'][use,branch]
    assert np.all(np.diff(keys)>0) and np.isfinite(values[qualified]).all()
    return keys,raw,prediction,status,qualified,values


def main():
    assert int(os.environ.get('PBS_NCPUS','0'))>0
    manifest=c.verify();assert not (c.ROOT/'sample.h5').exists()
    report=json.loads((c.CATALOGUE/'prepared.json').read_text())
    assert report['complete'] and c.sha(c.CATALOGUE/'catalogue.npz')==report['catalogue_sha256']
    with np.load(c.CATALOGUE/'catalogue.npz') as f:
        ap={k:f[k] for k in ('xy','lattice_index','stratum','inclusion_probability','population','allocation')}
        cached=f['predicted']
    A=len(ap['xy']);assert A==c.APERTURES and cached.shape==(A,6,4)
    # The cached molecule maps and the per-particle mass*abundance operator
    # agree independently on the completed pilot's 280 full-input projections.
    from plot_galic_selection import sample
    with h5py.File(c.PILOT/'sample.h5') as f:old_xy=f['apertures/xy'][:]
    with np.load(c.PILOT/'aperture_results.npz') as f:old_mol=f['full_proxy'][:,:,2:]
    for m,name in enumerate(c.MODELS):
        with np.load(c.parent.MAPS[m] if m<2 else c.parent.MAPS[2]) as f:
            labels=('C13O_abundance','C18O_abundance') if m<2 else (name+'_X13CO',name+'_XC18O')
            replay=np.column_stack([sample(f[k],old_xy) for k in labels])
        np.testing.assert_allclose(replay,old_mol[:,m],rtol=3e-7,atol=1e-20)
    keys,old_raw,old_pred,status,qualified,values=archive_pool()
    known=c.known_models(status,qualified)
    patches,bounds,reach=make_patches(ap['xy']);tree=cKDTree(ap['xy']*1000)
    registry=json.loads((c.ISO/'x8_full_direct_575_20261028/registry.json').read_text())
    with h5py.File(c.OLD/'census.h5') as f:factors=f['conversion_line'][:]
    baseline=np.zeros((A,6,8));baseline[:,:,:2]=baseline[:,:,2:4]=cached[:,:,:2]
    baseline[:,:,4:6]=baseline[:,:,6:8]=cached[:,:,2:]
    replay=np.zeros((A,6,2));counts=np.zeros(A,'i8');count=entries=0
    target=c.ROOT/'support.h5'
    fields=('actual','slug','mass','nuclei','xyz','h','row','particle_id','lum_per_H')
    with h5py.File(str(target)+'.tmp','w') as out:
        out.attrs.update(complete=False,manifest_sha256=manifest)
        for name,value in ap.items():out.create_dataset('apertures/'+name,data=value)
        for name,dtype in (('particle','i8'),('aperture','i8'),('weight','f8')):
            shape=(0,2) if name=='weight' else (0,)
            out.create_dataset('operator/'+name,shape=shape,maxshape=(None,)+shape[1:],dtype=dtype,chunks=True)
        for part,entry in enumerate(registry['parts']):
            with h5py.File(entry['path']) as f:
                assert f.attrs['complete']
                for start in range(0,len(f['mass']),100000):
                    stop=min(start+100000,len(f['mass']))
                    xyz,h=f['xyz'][start:stop],f['h'][start:stop]
                    candidates=tree.query_ball_point(xyz[:,:2],np.maximum(h,40)+reach+1e-6,workers=8)
                    sizes=np.fromiter((len(x) for x in candidates),dtype='i8',count=len(candidates))
                    if not sizes.sum():continue
                    particle=np.repeat(np.arange(len(h)),sizes)
                    aperture=np.concatenate([x for x in candidates if len(x)]).astype('i8')
                    wk=evaluate(xyz,h,particle,aperture,patches,bounds);nonzero=np.any(wk>0,axis=1)
                    particle,aperture,wk=particle[nonzero],aperture[nonzero],wk[nonzero]
                    if not len(particle):continue
                    unique,inverse=np.unique(particle,return_inverse=True)
                    block={name:f[name][start:stop][unique] for name in fields}
                    block.update(part=np.full(len(unique),part),cache_row=unique+start)
                    bk=part*(1<<32)+block['cache_row'];loc=np.searchsorted(keys,bk)
                    hit=loc<len(keys);hit[hit]&=keys[loc[hit]]==bk[hit]
                    oldloc=np.where(hit,loc,-1);block['archive_index']=oldloc
                    for name in ('actual','slug','particle_id'):
                        np.testing.assert_array_equal(block[name][hit],old_raw[name][loc[hit]])
                    for name,value in block.items():
                        dsname='particles/'+name
                        if dsname not in out:out.create_dataset(dsname,shape=(0,)+value.shape[1:],maxshape=(None,)+value.shape[1:],dtype=value.dtype,chunks=True)
                        ds=out[dsname];ds.resize(count+len(unique),axis=0);ds[count:]=value
                    for name,value in dict(particle=inverse+count,aperture=aperture,weight=wk).items():
                        ds=out['operator/'+name];ds.resize(entries+len(aperture),axis=0);ds[entries:]=value
                    for m in range(6):
                        op=int(m>=2);matrix=coo_matrix((wk[:,op],(aperture,inverse)),shape=(A,len(unique))).tocsr()
                        pline=block['nuclei'][:,None]*block['lum_per_H'][:,m,1:3]*factors
                        replay[:,m]+=matrix@pline
                        active=np.flatnonzero(hit)
                        active=active[known[loc[active],m]]
                        if not len(active):continue
                        old=loc[active];branch=c.MODEL_BRANCH[m]
                        q=qualified[old,branch] if m<2 else qualified[old][:,[0,2,3,4]].all(axis=1)
                        pl=pline[active];pm=block['mass'][active,None]*old_pred[old,m,1:3]
                        dl=block['nuclei'][active,None]*np.where(q[:,None],values[old,branch,4:6],0.)*factors
                        dm=block['mass'][active,None]*np.where(q[:,None],values[old,branch,1:3],0.)
                        residual=np.column_stack((dl-pl,np.where(q[:,None],0.,-pl),dm-pm,np.where(q[:,None],0.,-pm)))
                        baseline[:,m]+=matrix[:,active]@residual
                    counts+=np.bincount(aperture,minlength=A)
                    count+=len(unique);entries+=len(aperture)
            out.flush();print('support cache',part,'particles',count,'pairs',entries,flush=True)
        np.testing.assert_allclose(replay,cached[:,:,:2],rtol=3e-7,atol=1e-20)
        particle,aperture,wk=[out['operator/'+k][:] for k in ('particle','aperture','weight')]
        oldloc=out['particles/archive_index'][:]
        model_known=np.zeros((count,6),bool);hit=oldloc>=0;model_known[hit]=known[oldloc[hit]]
        density=np.digitize(np.log10(out['particles/actual'][:,0]),[1,3])
        nuclei=out['particles/nuclei'][:];light=out['particles/lum_per_H'][:,:,1:3]
        order=np.argsort(aperture,kind='stable');bounds2=np.searchsorted(aperture[order],np.arange(A+1))
        rng=np.random.default_rng(c.SEED);records=[];N=np.zeros((A,16),'i8');nh=N.copy()
        for a in range(A):
            ix=order[bounds2[a]:bounds2[a+1]];ids=particle[ix];weights=wk[ix]
            take=(~model_known[ids]).any(axis=1);ids,weights=ids[take],weights[take]
            assert len(np.unique(ids))==len(ids)
            if not len(ids):continue
            importance=np.zeros(len(ids))
            for m in range(6):
                contribution=weights[:,int(m>=2),None]*nuclei[ids,None]*light[ids,m]*factors
                score=np.max(contribution/np.maximum(cached[a,m,:2],1e-300),axis=1)
                importance=np.maximum(importance,np.where(model_known[ids,m],0.,score))
            rank=np.argsort(-importance,kind='stable');top=rank[:min(c.TOP,len(ids))];tail=rank[len(top):]
            N[a,15]=nh[a,15]=len(top);chosen=[top];labels=[np.full(len(top),15,'i8')]
            if len(tail):
                bins=np.digitize(np.log10(np.maximum(importance[tail],1e-300)),[-10,-8,-6,-4])
                strata=density[ids[tail]]*5+bins
                N[a,:15]=np.bincount(strata,minlength=15)
                scores=np.sqrt(N[a,:15]*np.bincount(strata,weights=importance[tail]**2,minlength=15))
                nh[a,:15]=allocate(N[a,:15],scores,min(c.TAIL,len(tail)))
                for h2 in np.flatnonzero(N[a,:15]):
                    members=tail[strata==h2];pick=rng.choice(members,int(nh[a,h2]),replace=False)
                    chosen.append(pick);labels.append(np.full(len(pick),h2,'i8'))
            pick,sh=np.concatenate(chosen),np.concatenate(labels)
            assert np.all((N[a]==0)|(nh[a]>=np.minimum(N[a],2)))
            records.append(dict(support_row=ids[pick],aperture=np.full(len(pick),a),stratum=sh,
                pi=nh[a,sh]/N[a,sh],kernel=weights[pick],unknown=~model_known[ids[pick]]))
        if records:
            records={k:np.concatenate([x[k] for x in records]) for k in records[0]}
        else:
            records=dict(support_row=np.empty(0,'i8'),aperture=np.empty(0,'i8'),stratum=np.empty(0,'i8'),
                pi=np.empty(0),kernel=np.empty((0,2)),unknown=np.empty((0,6),bool))
        unique,inverse=np.unique(records['support_row'],return_inverse=True);records['particle']=inverse
        assert len(unique)<=A*(c.TOP+c.TAIL)
        rows={k:read_rows(out['particles/'+k],unique) for k in ('actual','slug','mass','nuclei','particle_id','part','cache_row','archive_index','lum_per_H')}
        raw=(rows['actual'],rows['slug'],c.base.uniform(rows['actual'],registry['uniform_ratios']))
        spred=np.full((len(unique),6,6),np.nan);old=rows['archive_index'];found=old>=0
        spred[found]=old_pred[old[found]]
        if (~found).any():spred[~found]=c.parent.predict(tuple(x[~found] for x in raw))
        np.testing.assert_allclose(spred[:,:,3:],rows['lum_per_H'],rtol=2e-12,atol=1e-60)
        ss=np.zeros((len(unique),5),'i1');sq=np.zeros((len(unique),5),bool);sv=np.full((len(unique),5,6),np.nan)
        ss[found],sq[found],sv[found]=status[old[found]],qualified[old[found]],values[old[found]]
        needed=c.needed_states(ss,sq);tasks=np.flatnonzero(needed.ravel())
        assert len(tasks)<=5*A*(c.TOP+c.TAIL)
        sample=c.ROOT/'sample.h5'
        with h5py.File(str(sample)+'.tmp','w') as f:
            f.attrs.update(complete=False,manifest_sha256=manifest)
            for name,value in ap.items():f.create_dataset('apertures/'+name,data=value)
            for name,value in records.items():f.create_dataset('records/'+name,data=value)
            for name,value in dict(population=N,allocation=nh,baseline=baseline,full_proxy=cached,conversion_line=factors,tasks=tasks).items():f.create_dataset(name,data=value)
            for name,value in dict(actual=raw[0],slug=raw[1],uniform=raw[2],mass=rows['mass'],nuclei=rows['nuclei'],
                particle_id=rows['particle_id'],part=rows['part'],cache_row=rows['cache_row'],support_row=unique,
                prediction=spred,status=ss,qualified=sq,values=sv,needed=needed).items():f.create_dataset('sample/'+name,data=value,compression='gzip')
            f.attrs['complete']=True
        os.replace(str(sample)+'.tmp',sample)
        out.attrs['complete']=True
    os.replace(str(target)+'.tmp',target)
    c.save(c.ROOT/'prepared.json',dict(complete=True,manifest_sha256=manifest,sample_sha256=c.sha(sample),
        aperture_count=A,support_particles=count,support_pairs=entries,particle_records=len(inverse),
        distinct_reference_particles=len(unique),new_states=len(tasks),maximum_new_states=5*A*(c.TOP+c.TAIL),
        archive_attempted_states=int((status!=0).sum()),archive_qualified_states=int(qualified.sum()),
        selected_reused_states=int((ss!=0).sum()),selected_reused_qualified_states=int(sq.sum()),
        intentionally_unneeded_unsolved_states=int(((ss==0)&(~needed)).sum()),full_line_proxy_replay_passed=True,
        independent_cached_molecular_operator_check_passed=True,
        scope='Independent 606-aperture probability cohort; exact known labels plus sampled residuals; no full chemical census or model fitting.'))
    print('Prepared',A,'apertures;',len(unique),'selected particles;',len(tasks),'new states.',flush=True)


if __name__=='__main__':main()

"""Immutable full-population task identity and atomic result storage."""
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

ROOT=Path(__file__).resolve().parent
ISO=ROOT.parent
BRANCHES=('actual_on','slug_on','actual_off','uniform_on','uniform_off')
BASE_NAMES=('Tg_K','Td_K','X_H2','X_CO','X_13CO','X_C18O',
    'L_CO_1_0','L_13CO_1_0','L_C18O_1_0','tau_CO_1_0','tau_13CO_1_0','tau_C18O_1_0',
    'Tex_CO_1_0_K','Tex_13CO_1_0_K','Tex_C18O_1_0_K',
    'TB_CO_1_0_K_km_s','TB_13CO_1_0_K_km_s','TB_C18O_1_0_K_km_s')
OUTPUT_NAMES=BASE_NAMES+('beta_CO_1_0','beta_13CO_1_0','beta_C18O_1_0',
    'collider_xHI','collider_xHplus','collider_xe','collider_xpH2','collider_xoH2','collider_xHe')
SPECIES=('H2','H+','H2+','H3+','He+','O+','C+','CO','HCO+','Si+','CHx','OHx',
    '13C+','13CO','H13CO+','13CHx','18O+','C18O','HC18O+','OH18x')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(8*1024*1024): h.update(b)
    return h.hexdigest()


def save_json(path,value):
    path=Path(path); path.parent.mkdir(exist_ok=True,parents=True)
    temporary=Path(str(path)+'.tmp')
    with temporary.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temporary,path)


def verify_code():
    m=json.loads((ROOT/'manifest.json').read_text())
    for path,h in m['hashes'].items(): assert digest(path)==h,path
    for entry in m['cache_identities']+m['snapshot_identities']:
        s=Path(entry['path']).stat()
        assert s.st_size==entry['size'] and s.st_mtime_ns==entry['mtime_ns'],entry['path']
    return digest(ROOT/'manifest.json')


def load_plan():
    return json.loads((ROOT/'registry.json').read_text())


def split_group(xyz):
    x=np.floor(xyz[:,0]/1000).astype('i8'); y=np.floor(xyz[:,1]/1000).astype('i8')
    key=(x+2**31).astype('u8')^((y+2**31).astype('u8')<<np.uint64(32))^np.uint64(20261028)
    key=(key^(key>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
    key=(key^(key>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
    bucket=(key^(key>>np.uint64(31)))%100
    return np.where(bucket<70,0,np.where(bucket<85,1,2)).astype('i1')


def read_task(task,plan):
    entry=plan['parts'][task['part']]
    s=Path(entry['path']).stat()
    assert (s.st_size,s.st_mtime_ns)==(entry['size'],entry['mtime_ns'])
    sl=slice(task['start'],task['stop'])
    with h5py.File(entry['path'],'r') as f:
        assert f.attrs['complete'] and f.attrs['manifest_sha256']==plan['parent_manifest_sha256']
        names=('actual','slug','mass','xyz','h','row','particle_id')
        data={k:f[k][sl] for k in names}
    branch=task['branch']
    raw=data['slug'].copy() if branch=='slug_on' else data['actual'].copy()
    if branch.startswith('uniform'):
        for a,b,r in ((3,4,plan['uniform_ratios'][0]),(5,6,plan['uniform_ratios'][1])):
            total=raw[:,a]+raw[:,b]; raw[:,b]=total/(1+r); raw[:,a]=total-raw[:,b]
            np.testing.assert_allclose(raw[:,a]+raw[:,b],total,rtol=1e-14)
    assert len(raw)==task['stop']-task['start']
    assert np.all(np.isfinite(raw)) and np.all(raw[:,[0,1,2,3,5]]>0) and np.all(raw[:,[4,6]]>=0)
    data['raw']=raw; data['split']=split_group(data['xyz'])
    return data


def identity(task,data,plan,audit):
    h=hashlib.sha256()
    for name in ('raw','row','particle_id','split'):
        a=np.ascontiguousarray(data[name]); h.update(name.encode()); h.update(a.dtype.str.encode()); h.update(a.tobytes())
    return dict(task=task,block_input_sha256=h.hexdigest(),input_file_sha256=audit['cache_sha256'][plan['parts'][task['part']]['path']],
        manifest_sha256=audit['manifest_sha256'])


def paths(task,root=ROOT):
    suffix=Path(task['branch'])/f"part{task['part']:02d}"/f"rows_{task['start']:09d}_{task['stop']:09d}.npz"
    return root/'output'/suffix,root/'checkpoints'/suffix


def empty_state(n):
    return dict(status=np.zeros(n,'i1'),values=np.full((n,len(OUTPUT_NAMES)),np.nan),
        chemistry=np.full((n,len(SPECIES)),np.nan),attempts=np.zeros(n,'u1'),seconds=np.full(n,np.nan),
        ode_calls=np.zeros(n,'i4'),ode_failures=np.zeros(n,'i4'),quality_flags=np.zeros(n,'u2'),
        error=np.full(n,b'',dtype='S768'))


def atomic_state(path,metadata,state,data,complete):
    path=Path(path); path.parent.mkdir(exist_ok=True,parents=True)
    temporary=Path(str(path)+'.tmp.'+str(os.getpid()))
    with temporary.open('wb') as f:
        np.savez_compressed(f,metadata=np.asarray(json.dumps(metadata,sort_keys=True)),complete=np.asarray(complete),
            cache_row=np.arange(metadata['task']['start'],metadata['task']['stop'],dtype='i8'),
            snapshot_row=data['row'],particle_id=data['particle_id'],split=data['split'],**state)
        f.flush(); os.fsync(f.fileno())
    os.replace(temporary,path)


def read_state(path,metadata,n,require_complete=False,data=None):
    with np.load(path,allow_pickle=False) as f:
        assert json.loads(str(f['metadata']))==metadata,'Result identity mismatch'
        state={k:f[k].copy() for k in empty_state(n)}
        assert np.array_equal(f['cache_row'],np.arange(metadata['task']['start'],metadata['task']['stop']))
        if data is not None:
            assert np.array_equal(f['snapshot_row'],data['row'])
            assert np.array_equal(f['particle_id'],data['particle_id'])
            assert np.array_equal(f['split'],data['split'])
        assert state['values'].shape==(n,len(OUTPUT_NAMES)) and state['chemistry'].shape==(n,len(SPECIES))
        assert all(len(v)==n for v in state.values())
        assert np.all(np.isin(state['status'],[0,1,-1,-2]))
        assert np.all(np.isfinite(state['values'][state['status']==1]))
        assert np.all(np.isnan(state['values'][state['status']<=0]))
        if require_complete: assert bool(f['complete']) and np.all(state['status']!=0)
    return state

"""Checkpointed five-branch direct solves using the accepted extended worker."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import multiprocessing
import os
import time

import h5py
import numpy as np

import outer_common as c
import solve_diagnostics as worker
import engine


def main():
    assert int(os.environ.get('PBS_NCPUS', '0')) > 0
    sha, ready = c.ready()
    part = int(os.environ['PARTITION'])
    assert 0 <= part < c.PARTITIONS
    with h5py.File(c.ROOT/'sample.h5') as f:
        n = len(f['sample/mass'])
        ids = np.arange(part, n*5, c.PARTITIONS)
        uid, branch = ids//5, ids % 5
        raw = {k: f['sample/'+k][:] for k in ('actual', 'slug', 'uniform')}
        particle_ids = f['sample/particle_id'][:][uid]
        reused = f['reuse_qualified'][:][uid, branch]
        reused_values = f['reuse_values'][:][uid, branch]
    target = c.ROOT/f'direct/part{part:02d}.npz'
    lock = (c.ROOT/f'direct/part{part:02d}.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    meta = dict(manifest_sha256=sha, sample_sha256=ready['sample_sha256'], partition=part,
                protocol='mxstep5000_600s', tEqGuess=None, smallabd=1e-15)
    state = c.base.core.empty_state(len(ids))
    final = np.full((len(ids), 3), -1, 'i1')
    origin = np.zeros(len(ids), 'u1')
    if target.exists():
        with np.load(target) as f:
            assert json.loads(str(f['metadata'])) == meta
            np.testing.assert_array_equal(f['task_ids'], ids)
            np.testing.assert_array_equal(f['particle_id'], particle_ids)
            state = {k: f[k].copy() for k in state}
            final, origin = f['final_returns'].copy(), f['origin'].copy()
    else:
        state['status'][reused] = 1
        state['values'][reused] = reused_values[reused]
        final[reused] = 1
        origin[reused] = 1
    started = time.monotonic()

    def checkpoint():
        q = (state['status'] == 1) & np.all(final == 1, axis=1)
        complete = bool(np.all(state['status'] != 0))
        c.base.atomic_npz(target, metadata=np.asarray(json.dumps(meta, sort_keys=True)),
            task_ids=ids, particle_id=particle_ids, final_returns=final, origin=origin,
            qualified=q, complete=np.asarray(complete), **state)
        c.save(c.ROOT/f'direct/part{part:02d}.json', dict(partition=part, complete=complete,
            total=len(ids), pending=int((state['status'] == 0).sum()), qualified=int(q.sum()),
            failed=int((state['status'] == -1).sum()), timeout=int((state['status'] == -2).sum()),
            reused=int((origin == 1).sum()), elapsed_seconds=time.monotonic()-started))

    checkpoint()
    try:
        with ProcessPoolExecutor(max_workers=min(104, int(os.environ['PBS_NCPUS'])),
                mp_context=multiprocessing.get_context('spawn'), initializer=worker.initialize) as pool:
            futures = {}
            for i in np.flatnonzero(state['status'] == 0):
                name = c.BRANCHES[branch[i]]
                key = 'slug' if name == 'slug_on' else 'uniform' if name.startswith('uniform') else 'actual'
                futures[pool.submit(worker.solve, (int(i), 1, name, tuple(raw[key][uid[i]])))] = int(i)
            for done, future in enumerate(as_completed(futures), 1):
                result, returns, damping, events, trace = future.result()
                i = futures[future]
                assert result[0] == i
                engine.apply_result(state, result)
                final[i], origin[i] = returns, 2
                if done % 128 == 0:
                    checkpoint()
                    print('outer direct', part, done, '/', len(futures), flush=True)
    finally:
        checkpoint()
        lock.close()
    assert np.all(state['status'] != 0), 'Pending states require explicit resume'


if __name__ == '__main__':
    main()

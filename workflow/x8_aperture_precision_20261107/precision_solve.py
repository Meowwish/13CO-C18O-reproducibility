"""Small, checkpointed partitions of only the selected unresolved states."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
import multiprocessing
import os
import time

import h5py
import numpy as np

import precision_common as c
import solve_diagnostics as worker
import engine


def read_rows(dataset, indices):
    result = np.empty((len(indices),)+dataset.shape[1:], dtype=dataset.dtype)
    for start in range(0, len(dataset), 100000):
        lo, hi = np.searchsorted(indices, [start, start+100000])
        if hi > lo: result[lo:hi] = dataset[start:start+100000][indices[lo:hi]-start]
    return result


def main():
    assert int(os.environ.get('PBS_NCPUS', '0')) > 0
    sha, ready = c.ready(); part = int(os.environ['PARTITION'])
    assert 0 <= part < c.PARTITIONS
    with h5py.File(c.ROOT/'sample.h5') as f:
        ids = f['tasks'][part::c.PARTITIONS]
        uid, branch = ids//5, ids % 5
        unique, inverse = np.unique(uid, return_inverse=True)
        raw = {key: read_rows(f['sample/'+key], unique) for key in ('actual', 'slug', 'uniform')}
        particle_ids = read_rows(f['sample/particle_id'], unique)[inverse]
        assert np.all(read_rows(f['sample/status'], unique)[inverse, branch] == 0)
    target = c.ROOT/f'direct/part{part:02d}.npz'
    lock = (c.ROOT/f'direct/part{part:02d}.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    metadata = dict(manifest_sha256=sha, sample_sha256=ready['sample_sha256'], partition=part,
        protocol='mxstep5000_600s', tEqGuess=None, smallabd=1e-15)
    state = c.base.core.empty_state(len(ids)); final = np.full((len(ids), 3), -1, 'i1')
    if target.exists():
        with np.load(target) as f:
            assert json.loads(str(f['metadata'])) == metadata
            np.testing.assert_array_equal(f['task_ids'], ids)
            np.testing.assert_array_equal(f['particle_id'], particle_ids)
            state = {key: f[key].copy() for key in state}; final = f['final_returns'].copy()
    started = time.monotonic()

    def checkpoint():
        q = (state['status'] == 1)&np.all(final == 1, axis=1)
        complete = bool(np.all(state['status'] != 0))
        c.base.atomic_npz(target, metadata=np.asarray(json.dumps(metadata, sort_keys=True)), task_ids=ids,
            particle_id=particle_ids, final_returns=final, qualified=q, complete=np.asarray(complete), **state)
        c.save(c.ROOT/f'direct/part{part:02d}.json', dict(partition=part, total=len(ids), complete=complete,
            pending=int((state['status'] == 0).sum()), qualified=int(q.sum()),
            failed=int((state['status'] == -1).sum()), timeout=int((state['status'] == -2).sum()),
            elapsed_seconds=time.monotonic()-started))

    checkpoint()
    try:
        with ProcessPoolExecutor(max_workers=min(104, int(os.environ['PBS_NCPUS'])),
                mp_context=multiprocessing.get_context('spawn'), initializer=worker.initialize) as pool:
            pending = iter(np.flatnonzero(state['status'] == 0))
            futures = {}

            def enqueue():
                i = next(pending, None)
                if i is None: return False
                name = c.BRANCHES[branch[i]]
                key = 'slug' if name == 'slug_on' else 'uniform' if name.startswith('uniform') else 'actual'
                futures[pool.submit(worker.solve, (int(i), 1, name, tuple(raw[key][inverse[i]])))] = int(i)
                return True

            for _ in range(min(416, len(ids))):
                if not enqueue(): break
            done_count = 0
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    i = futures.pop(future)
                    result, returns, damping, events, trace = future.result()
                    assert result[0] == i
                    engine.apply_result(state, result); final[i] = returns
                    done_count += 1; enqueue()
                    if done_count % 512 == 0:
                        checkpoint()
                        print('precision direct', part, int((state['status'] != 0).sum()), '/', len(ids), flush=True)
    finally:
        checkpoint(); lock.close()
    assert np.all(state['status'] != 0), 'Pending states require continuation'


if __name__ == '__main__':
    main()

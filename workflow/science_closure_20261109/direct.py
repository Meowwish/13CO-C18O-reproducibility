"""Bounded checkpointed work queue; no change to physical parameters."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
import multiprocessing
import os
import time

import h5py
import numpy as np

import closure_common as c
import solve_diagnostics as worker
import engine


def initialize(recovery):
    # The diagnostic worker exposes the ODE work limit and whole-state alarm.
    # Changing these module constants leaves historical files/checkpoints intact.
    if recovery:
        worker.c.MXSTEP = 50000
        worker.c.WALL_SECONDS = 1800
    worker.initialize()


def main():
    assert int(os.environ.get('PBS_NCPUS', '0')) > 0
    manifest = c.verify(); batch = os.environ['BATCH']; part = int(os.environ['PARTITION'])
    assert batch in ('recovery', *[f'temporal{x}' for x in c.SNAPSHOTS]) and 0 <= part < c.PARTITIONS
    folder = c.ROOT/batch; ready = json.loads((folder/'prepared.json').read_text())
    assert ready['manifest_sha256'] == manifest and c.sha(folder/'cases.h5') == ready['cases_sha256']
    with h5py.File(folder/'cases.h5') as f:
        ids = np.arange(part, len(f['raw']), c.PARTITIONS)
        raw, branches, particle = (f[k][:][ids] for k in ('raw', 'branch', 'particle_id'))
    out = folder/'direct'; out.mkdir(exist_ok=True)
    lock = (out/f'part{part:02d}.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    meta = dict(manifest_sha256=manifest, cases_sha256=ready['cases_sha256'], batch=batch, partition=part,
        tEqGuess=None, smallabd=1e-15, tol=.001, mxstep=50000 if batch == 'recovery' else 5000,
        wall_seconds=1800 if batch == 'recovery' else 600, maxTime_s=1e18)
    state = c.parent.base.core.empty_state(len(ids)); final = np.full((len(ids), 3), -1, 'i1')
    target = out/f'part{part:02d}.npz'
    if target.exists():
        with np.load(target) as f:
            assert json.loads(str(f['metadata'])) == meta
            np.testing.assert_array_equal(f['case_ids'], ids)
            state = {k:f[k].copy() for k in state}; final = f['final_returns'].copy()
    started = time.monotonic()
    def checkpoint():
        q = (state['status'] == 1) & np.all(final == 1, axis=1)
        done = bool(np.all(state['status'] != 0))
        c.parent.base.atomic_npz(target, metadata=np.asarray(json.dumps(meta, sort_keys=True)),
            case_ids=ids, particle_id=particle, final_returns=final, qualified=q, complete=np.asarray(done), **state)
        c.save(out/f'part{part:02d}.json', dict(complete=done, total=len(ids),
            pending=int((state['status'] == 0).sum()), qualified=int(q.sum()),
            failed=int((state['status'] == -1).sum()), timeout=int((state['status'] == -2).sum()),
            elapsed_seconds=time.monotonic()-started))
    checkpoint()
    try:
        with ProcessPoolExecutor(max_workers=min(104, int(os.environ['PBS_NCPUS'])),
                mp_context=multiprocessing.get_context('spawn'), initializer=initialize,
                initargs=(batch == 'recovery',)) as pool:
            todo = iter(np.flatnonzero(state['status'] == 0)); active = {}
            def enqueue():
                i = next(todo, None)
                if i is None:return False
                active[pool.submit(worker.solve, (int(i), 1, c.BRANCHES[branches[i]], tuple(raw[i])))] = int(i)
                return True
            for _ in range(min(416, len(ids))):
                if not enqueue():break
            done_count = 0
            while active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    i = active.pop(future)
                    result, returns, *_ = future.result()
                    assert result[0] == i
                    engine.apply_result(state, result); final[i] = returns
                    done_count += 1; enqueue()
                    if done_count % 64 == 0:checkpoint()
    finally:
        checkpoint(); lock.close()
    assert np.all(state['status'] != 0), 'Continuation required; no completed scientific inference.'


if __name__ == '__main__':main()

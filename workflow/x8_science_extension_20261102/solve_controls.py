"""PBS-only controls using the existing strict resource-extended runtime."""
import copy
import fcntl
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import h5py
import numpy as np
import extension as c
import solve_diagnostics as runtime
import engine

BASE_CLOUD = None


def initialize():
    global BASE_CLOUD
    runtime.initialize()
    BASE_CLOUD = copy.deepcopy(runtime.phase.worker.DIRECT.CLOUD)


def solve(task):
    index, scenario, branch, raw = task
    _, uv, cr, _ = c.SCENARIOS[scenario]
    cloud = copy.deepcopy(BASE_CLOUD)
    cloud['chi_Draine'] *= uv
    cloud['cosmic_ray_ionization_rate_s-1'] *= cr
    runtime.phase.worker.DIRECT.CLOUD = cloud
    return runtime.solve((index, 1, 'control_on' if branch == 0 else 'control_off', raw))


def main():
    assert int(os.environ.get('PBS_NCPUS', '0')) > 0
    sha = c.verify(); part = int(os.environ['PARTITION'])
    assert 0 <= part < 8
    with h5py.File(c.ROOT/'cases.h5') as f:
        ids = np.arange(part, len(f['sample_id']), 8)
        raw, scenario, branch = (f[k][:][ids] for k in ('raw','scenario','branch'))
    target = c.ROOT/f'direct/part{part:02d}.npz'
    lock = target.with_suffix('.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    state = c.design.core.empty_state(len(ids))
    final = np.full((len(ids), 3), -1, 'i1')
    damping = np.full((len(ids), 3), np.nan)
    events = np.full(len(ids), b'', 'S4096')
    traces = events.copy()
    if target.exists():
        with np.load(target) as f:
            assert str(f['manifest_sha256']) == sha
            np.testing.assert_array_equal(f['case_ids'], ids)
            for key in state: state[key] = f[key].copy()
            final, damping, events, traces = (f[k].copy() for k in ('final_returns','damping','events','trace'))

    def checkpoint():
        complete = bool(np.all(state['status'] != 0))
        c.design.atomic_npz(target, manifest_sha256=np.asarray(sha), case_ids=ids,
                           complete=np.asarray(complete), final_returns=final, damping=damping,
                           events=events, trace=traces, **state)
        c.save(target.with_suffix('.json'), dict(complete=complete, total=len(ids),
               pending=int((state['status']==0).sum()), success=int((state['status']==1).sum()),
               failed=int((state['status']<0).sum())))
    try:
        with ProcessPoolExecutor(max_workers=min(104, int(os.environ['PBS_NCPUS'])),
                mp_context=multiprocessing.get_context('spawn'), initializer=initialize) as pool:
            futures = {pool.submit(solve, (int(i), int(scenario[i]), int(branch[i]), tuple(raw[i]))): int(i)
                       for i in np.flatnonzero(state['status'] == 0)}
            for count, future in enumerate(as_completed(futures), 1):
                result, returns, df, ev, trace = future.result()
                i = futures[future]; assert result[0] == i
                engine.apply_result(state, result)
                final[i], damping[i], events[i], traces[i] = returns, df, ev.encode(), trace.encode()
                if count % 128 == 0:
                    checkpoint(); print('partition', part, 'completed', count, flush=True)
    finally:
        checkpoint(); lock.close()
    assert np.all(state['status'] != 0)


if __name__ == '__main__': main()

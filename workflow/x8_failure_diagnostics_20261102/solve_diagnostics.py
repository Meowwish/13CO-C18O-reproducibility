"""One fixed pair of numerical-resource experiments on existing states."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import multiprocessing
import os
import signal
import time
from types import SimpleNamespace

import h5py
import numpy as np

import diagnostics_common as c
import run_cases as phase
import engine

REAL_ALARM = signal.alarm
BACKEND = None
PROTOCOL = 0
TRACE = None


def wall_alarm(seconds):
    assert seconds in (0, 180)
    return REAL_ALARM(c.WALL_SECONDS if seconds else 0)


def timeout_handler(*_):
    raise phase.worker.CaseTimeout(f'{c.WALL_SECONDS} s total physical-state wall budget exceeded')


def budgeted_odeint(*args, **kwargs):
    assert 'mxstep' not in kwargs, 'Unexpected upstream ODE override'
    if PROTOCOL == 1:
        kwargs['mxstep'] = c.MXSTEP
    started = time.monotonic()
    TRACE['calls'] += 1
    try:
        result = BACKEND(*args, **kwargs)
    except BaseException as exc:
        TRACE['last_exception'] = (type(exc).__name__ + ': ' + str(exc))[:400]
        raise
    else:
        if kwargs.get('full_output', False):
            _, info = result
            message = str(info.get('message'))
            TRACE['messages'][message] = TRACE['messages'].get(message, 0) + 1
            # Failure leaves some scipy diagnostic-array entries undefined.
            if message == 'Integration successful.':
                for key in ('nst', 'nfe', 'nje'):
                    a = np.asarray(info.get(key, []))
                    if a.size:
                        TRACE['max_' + key] = max(TRACE['max_' + key], int(np.max(a)))
            if message != 'Integration successful.':
                grid = np.asarray(args[2])
                TRACE['last_rejected_interval_s'] = [float(grid[0]), float(grid[-1])]
        return result
    finally:
        TRACE['ode_wall_seconds'] += time.monotonic() - started


def initialize():
    global BACKEND
    phase.initialize()
    assert phase.BASE_SOLVER['tEqGuess_s'] is None
    BACKEND = phase.worker.ORIGINAL_ODEINT
    phase.worker.ORIGINAL_ODEINT = budgeted_odeint
    # Substitute only the worker's alarm interface, not Python's signal module.
    phase.worker.signal = SimpleNamespace(alarm=wall_alarm)
    signal.signal(signal.SIGALRM, timeout_handler)


def solve(task):
    global PROTOCOL, TRACE
    index, protocol, branch, raw = task
    PROTOCOL = int(protocol)
    assert PROTOCOL in (0, 1)
    TRACE = dict(calls=0, messages={}, max_nst=0, max_nfe=0, max_nje=0, ode_wall_seconds=0.)
    # mode=0 in the existing phase wrapper forwards only maxTime=1e18.
    result, final, damping, events = phase.solve((index, 0, branch, raw))
    trace = json.dumps(TRACE, sort_keys=True)
    assert len(trace.encode()) < 4096
    return result, final, damping, events, trace


def main():
    assert int(os.environ.get('PBS_NCPUS', '0')) > 0
    sha = c.verify()
    part = int(os.environ['PARTITION'])
    assert 0 <= part < 8
    with h5py.File(c.ROOT / 'cases.h5') as f:
        ids = np.flatnonzero(f['partition'][:] == part)
        states = f['state_id'][:][ids]
        protocols = f['protocol'][:][ids]
        branches = f['branch'][:][ids]
        raw = f['raw'][:][ids]
    target = c.ROOT / f'output/part{part:02d}.npz'
    lock = (c.ROOT / f'output/part{part:02d}.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = c.base.core.empty_state(len(ids))
    final = np.full((len(ids), 3), -1, 'i1')
    damping = np.full((len(ids), 3), np.nan)
    events = np.full(len(ids), b'', 'S4096')
    traces = np.full(len(ids), b'', 'S4096')
    meta = dict(manifest_sha256=sha, partition=part, protocols=c.PROTOCOLS)
    if target.exists():
        with np.load(target) as f:
            assert json.loads(str(f['metadata'])) == json.loads(json.dumps(meta))
            np.testing.assert_array_equal(f['case_ids'], ids)
            for key in state:
                state[key] = f[key].copy()
            final, damping, events, traces = (f[key].copy() for key in ('final_returns', 'final_damping', 'events', 'ode_trace'))
    started = time.monotonic()

    def checkpoint():
        complete = bool(np.all(state['status'] != 0))
        c.base.atomic_npz(target, metadata=np.asarray(json.dumps(meta, sort_keys=True)), complete=np.asarray(complete),
                          case_ids=ids, state_id=states, protocol=protocols, final_returns=final,
                          final_damping=damping, events=events, ode_trace=traces, **state)
        c.save(c.ROOT / f'output/part{part:02d}.json', dict(partition=part, complete=complete, total=len(ids),
               pending=int((state['status'] == 0).sum()), success=int((state['status'] == 1).sum()),
               failed=int((state['status'] == -1).sum()), timeout=int((state['status'] == -2).sum()), elapsed_seconds=time.monotonic()-started))

    try:
        with ProcessPoolExecutor(max_workers=min(104, int(os.environ['PBS_NCPUS'])),
                mp_context=multiprocessing.get_context('spawn'), initializer=initialize) as pool:
            futures = {pool.submit(solve, (int(i), int(protocols[i]), c.base.BRANCHES[branches[i]], tuple(raw[i]))): int(i)
                       for i in np.flatnonzero(state['status'] == 0)}
            for count, future in enumerate(as_completed(futures), 1):
                result, returns, df, ev, trace = future.result()
                i = futures[future]
                assert result[0] == i
                engine.apply_result(state, result)
                final[i], damping[i], events[i], traces[i] = returns, df, ev.encode(), trace.encode()
                if count % 64 == 0:
                    checkpoint()
                    print('diagnostic partition', part, count, flush=True)
    finally:
        checkpoint()
        lock.close()
    assert np.all(state['status'] != 0)


if __name__ == '__main__':
    main()

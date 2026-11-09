"""Check resource forwarding and admission without executing chemistry."""
from collections import Counter
import copy
from types import SimpleNamespace

import h5py
import numpy as np

import diagnostics_common as c
import solve_diagnostics as d


def main():
    calls = []
    def backend(*args, **kwargs):
        calls.append((args, kwargs.copy()))
        return np.ones((2, 1)), dict(message='Integration successful.', nst=[1], nfe=[2], nje=[0])
    d.BACKEND = backend
    args = (object(), np.array([.1]), np.array([0., 1.]))
    options = dict(full_output=True, rtol=.001, atol=1e-15)
    for protocol in (0, 1):
        d.PROTOCOL = protocol
        d.TRACE = dict(calls=0, messages={}, max_nst=0, max_nfe=0, max_nje=0, ode_wall_seconds=0.)
        result = d.budgeted_odeint(*args, **options)
        assert result[1]['message'] == 'Integration successful.' and d.TRACE['calls'] == 1
        assert all(a is b for a, b in zip(calls[-1][0], args))
        expected = {**options, **({'mxstep': 5000} if protocol else {})}
        assert calls[-1][1] == expected
    d.BACKEND = lambda *a, **k: (_ for _ in ()).throw(ValueError('stub integration failure'))
    try:
        d.budgeted_odeint(*args, **options)
    except ValueError:
        pass
    else:
        raise AssertionError('ODE wrapper swallowed an exception')
    alarms = []
    d.REAL_ALARM = lambda seconds: alarms.append(seconds)
    d.wall_alarm(180)
    d.wall_alarm(0)
    assert alarms == [600, 0]
    try:
        d.timeout_handler()
    except d.phase.worker.CaseTimeout as exc:
        assert '600 s' in str(exc)
    else:
        raise AssertionError('Missing timeout')
    np.testing.assert_array_equal(c.admissible(np.array([1, 1, -1]), np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])), [True, False, False])
    assert np.all(c.agreement_error([1, -1, 0], [1, -1, 0]) == 0)
    # Test unchanged AUTO/tolerance/damping forwarding using a fake worker.
    phase = d.phase
    original = phase.worker.DIRECT, phase.worker.solve, phase.BASE_SOLVER
    solver = dict(tEqGuess_s=None, maximum_time_primary_s=1e16, maximum_time_retry_s=1e18, relative_tolerance=.001)
    phase.BASE_SOLVER = copy.deepcopy(solver)
    phase.worker.DIRECT = SimpleNamespace(SOLVER=copy.deepcopy(solver))
    seen = []
    def fake(task):
        seen.append(copy.deepcopy(phase.worker.DIRECT.SOLVER))
        return task[0], 1, 1, .1, 1, 0, 0, np.ones(27), np.ones(20), ''
    phase.worker.solve = fake
    try:
        for protocol in (0, 1):
            d.solve((protocol, protocol, 'actual_on', (1.,)*7))
    finally:
        phase.worker.DIRECT, phase.worker.solve, phase.BASE_SOLVER = original
    assert all(x == {**solver, 'maximum_time_primary_s': 1e18} for x in seen)
    phase.EVENTS = Counter(); phase.FINAL = {}; phase.FINAL_DAMPING = {}; phase.PHASE = 'capture'
    emitter = SimpleNamespace(name='13co', dampFactor=.25)
    assert phase.wrap_emitter(lambda *a, **k: False)(emitter) is False
    assert phase.FINAL['13co'] == 0
    assert phase.wrap_emitter(lambda *a, **k: True)(emitter) is True
    assert phase.FINAL['13co'] == 1
    with h5py.File(c.ROOT / 'cases.h5') as f:
        ids, modes, kind = (f[k][:] for k in ('state_id', 'protocol', 'kind'))
        np.testing.assert_array_equal(ids[::2], ids[1::2])
        np.testing.assert_array_equal(modes, np.tile([0, 1], len(ids)//2))
        assert np.sum(kind == 0) == 5450 and np.sum(kind == 1) == 320
        assert np.all(np.isfinite(f['raw'][:]))
        assert len(np.unique(np.column_stack((ids, modes)), axis=0)) == 5770
        for part in range(8):
            assert np.sum(f['partition'][:] == part) <= 722
    c.save(c.ROOT / 'local_checks.json', dict(passed=True, chemistry_executed=False, ODE_forwarding_preserved=True,
        default_work_limit_unchanged_in_A=True, only_mxstep_changed_in_B=True, AUTO_and_tolerance_preserved=True,
        wall_budget_and_cleanup_verified=True, exceptions_propagate=True, final_return_required=True,
        paired_case_identity_verified=True, fixed_case_count=5770))
    print('Resource-wrapper, AUTO, final-return and fixed-case checks passed; no chemistry executed.')


if __name__ == '__main__':
    main()

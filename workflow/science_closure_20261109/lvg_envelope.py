"""Conservative element-constrained envelopes for the implemented LVG law."""
from pathlib import Path
import numpy as np

PLANCK = 6.62607015e-27
LIGHT = 2.99792458e10
BOLTZMANN = 1.380649e-16
PC = 3.0856775814913673e18


def lamda_transition(path):
    lines = [x.strip() for x in Path(path).read_text().splitlines() if x.strip() and not x.lstrip().startswith('!')]
    levels = int(lines[2]); transitions = int(lines[3+levels]); assert transitions > 0
    t = lines[4+levels].split(); assert [int(x) for x in t[:3]] == [1, 2, 1]
    g0, g1 = float(lines[3].split()[2]), float(lines[4].split()[2])
    return float(t[3]), float(t[4])*1e9, g1/g0


def bounds(raw, transition, isotope, allow_inversion=True):
    """Per-H signed luminosity bounds, conditional on conserved elemental caps.

    For J=1->0, populations lie in [0,1] and sum to unity. In the
    implemented LVG solver, tau=K*x*(g*f0-f1). At negative tau it uses
    beta=1-tau/2, hence beta <= 1+K*x/2 even if inversion is allowed.
    This is a bound on this implementation, not an exact maser transfer law.
    """
    A, nu, g = transition
    raw = np.asarray(raw)
    cap = np.minimum(raw[:, 4], raw[:, 5]) if isotope == '13CO' else np.minimum(raw[:, 3], raw[:, 6])
    assert np.all(np.isfinite(raw)) and np.all(cap >= 0) and np.all(raw[:, 2] > 0)
    K = A*(LIGHT/nu)**3*raw[:, 0]/(8*np.pi*raw[:, 2]*1e5/PC)
    beta = 1+K*cap/2 if allow_inversion else np.ones(len(raw))
    photon = 1/np.expm1(PLANCK*nu/(BOLTZMANN*2.73))
    scale = A*PLANCK*nu*cap*beta
    return np.column_stack((-g*photon*scale, (1+photon)*scale))


def ratio_box(lo, hi):
    if np.any(lo <= 0) or np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)):
        return None
    return [float(np.log10(lo[0]/hi[1])), float(np.log10(hi[0]/lo[1]))]

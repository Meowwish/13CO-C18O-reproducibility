"""Positive rescaled stratum weights, including finite-population correction."""
import numpy as np


def replicate_weights(strata,population,rng,reps=800):
    strata=np.asarray(strata);out=np.ones((reps,len(strata)))
    for h in np.unique(strata):
        use=np.flatnonzero(strata==h);n=len(use);N=int(population[h])
        assert 0<n<=N
        if n==N:continue
        assert n>=2
        draws=rng.multinomial(n-1,np.full(n,1/n),size=reps)
        scale=np.sqrt(1-n/N)
        out[:,use]=(1-scale)+scale*n/(n-1)*draws
        np.testing.assert_allclose(out[:,use].sum(axis=1),n,rtol=1e-12)
        assert np.all(out[:,use]>0)
    return out

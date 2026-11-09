"""Design-based joint covariance and paired log-ratio delta intervals."""
import numpy as np


def joint_covariance(rows,strata,population,sample_n):
    """rows already include 1/pi; SRSWOR variance is n_h(1-f_h) S_rows,h."""
    rows=np.asarray(rows);out=np.zeros((rows.shape[1],rows.shape[1]))
    for h in np.unique(strata):
        group=rows[strata==h];n=len(group);N=int(population[h])
        assert n==sample_n[h] and 0<n<=N
        if n==N:continue
        assert n>=2,'Non-census singleton variance is not estimable'
        centred=group-group.mean(axis=0)
        out+=n*(1-n/N)*(centred.T@centred)/(n-1)
    return out


def interval(point,gradient,covariance):
    if not np.isfinite(point) or not np.all(np.isfinite(gradient)):
        return dict(point=None,standard_error=None,interval_95=None)
    variance=float(gradient@covariance@gradient)
    se=np.sqrt(max(0.,variance))
    return dict(point=float(point),standard_error=float(se),interval_95=[float(point-1.96*se),float(point+1.96*se)])


def log_metrics(direct,proxy,complete,frequency_term):
    """Jacobians with respect to the joint [sum(direct),sum(proxy)] vector."""
    k=len(direct);size=k*3;I=np.eye(size)
    dmap=np.column_stack((I,-I if complete else np.zeros_like(I)))
    pmap=np.zeros((size,2*size)) if complete else np.column_stack((np.zeros_like(I),I))
    with np.errstate(invalid='ignore',divide='ignore'):
        ld=np.log10(direct.ravel());lp=np.log10(proxy.ravel())
        dg=dmap/(np.log(10)*direct.ravel()[:,None])
        pg=pmap/(np.log(10)*proxy.ravel()[:,None])
    ld=np.where(direct.ravel()>0,ld,np.nan);lp=np.where(proxy.ravel()>0,lp,np.nan)
    rd=ld.reshape(k,3);rp=lp.reshape(k,3)
    rdg=dg.reshape(k,3,2*size);rpg=pg.reshape(k,3,2*size)
    return dict(line_error=ld-lp,line_error_gradient=dg-pg,
        direct_ratio=rd[:,1]-rd[:,2]+frequency_term,proxy_ratio=rp[:,1]-rp[:,2]+frequency_term,
        direct_ratio_gradient=rdg[:,1]-rdg[:,2],proxy_ratio_gradient=rpg[:,1]-rpg[:,2])


def ess(weights):
    weights=np.asarray(weights);denom=float(weights@weights)
    return float(weights.sum()**2/denom) if denom>0 else None


def weighted_quantiles(values,weights):
    good=np.isfinite(values)&np.isfinite(weights)&(weights>0)
    if not good.any():return None
    v,w=values[good],weights[good];order=np.argsort(v);v,w=v[order],w[order]
    return np.interp([.1,.5,.9,.95],np.cumsum(w)/w.sum(),v).tolist()

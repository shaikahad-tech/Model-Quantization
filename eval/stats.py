"""Paired statistics; arrays must refer to the same ordered cases."""
import math
import random
from statistics import NormalDist


def wilson_ci(k, n, alpha=0.05):
    if not isinstance(n,int) or not isinstance(k,int) or not 0 <= k <= n or n == 0 or not 0 < alpha < 1:
        raise ValueError('Expected 0 <= k <= n, n>0 and 0<alpha<1')
    z=NormalDist().inv_cdf(1-alpha/2);p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return (0. if k==0 else max(0.,center-radius)),(1. if k==n else min(1.,center+radius))


def paired_bootstrap_diff(a,b,n_boot=10000,seed=20261111,alpha=0.05):
    import numpy as np
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if a.ndim!=1 or a.shape!=b.shape or not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all() or n_boot<100 or not 0<alpha<1:
        raise ValueError('Finite paired nonempty vectors and at least 100 resamples required')
    d=a-b;rng=np.random.default_rng(seed);sample=[]
    for offset in range(0,n_boot,256):
        sample.extend(d[rng.integers(0,len(d),size=(min(256,n_boot-offset),len(d)))].mean(axis=1).tolist())
    low,high=np.quantile(sample,[alpha/2,1-alpha/2])
    return {'n':len(d),'point':float(d.mean()),'low':float(low),'high':float(high),'alpha':alpha,'n_boot':n_boot,'seed':seed}


def mcnemar_exact(b,c):
    if any(type(x) is not int or x<0 for x in (b,c)):raise ValueError('Discordant counts must be nonnegative integers')
    n=b+c
    if not n:return 1.0
    log_terms=[math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)-n*math.log(2) for i in range(min(b,c)+1)]
    return min(1.0,2*sum(math.exp(t) for t in log_terms))


def noninferior(a,b,delta,alpha=0.05,n_boot=10000,seed=20261111):
    if not 0<=delta<=1 or not 0<alpha<0.5:raise ValueError('Invalid margin/alpha')
    result=paired_bootstrap_diff(a,b,n_boot,seed,alpha=2*alpha)
    return dict(result,one_sided_alpha=alpha,margin=delta,passed=result['low']>=-delta)


def power_paired_binary(n,p_gain,p_loss,alpha=0.05):
    """Normal-approximate one-sided power; pilot discordance assumptions, not observed power."""
    if n<=0 or min(p_gain,p_loss)<0 or p_gain+p_loss>1 or not 0<alpha<0.5:raise ValueError('Invalid paired design')
    delta=p_gain-p_loss;variance=p_gain+p_loss-delta*delta
    if variance==0:return float(delta>0)
    z=NormalDist().inv_cdf(1-alpha)
    return NormalDist().cdf(delta*math.sqrt(n/variance)-z)

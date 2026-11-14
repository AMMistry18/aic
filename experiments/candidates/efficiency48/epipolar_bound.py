"""Admissible lower bound for original weighted-L1 fixture reprojection gate.

For one correspondence, S=w1*|delta1|_1+w2*|delta2|_1 must satisfy
|x2' F x1| <= a*S+b*S^2. Solving gives a lower bound on any feasible
reprojection sum, including nonlinear/nonplanar fits. A point shared by k
views appears in at most k-1 pair bounds, so divide summed pair bounds by
k-1. Never replace the original fit or accept a pose from this bound.
"""
import itertools
import numpy as np


def fundamental(first,second):
    A=np.asarray(first['T']);B=np.asarray(second['T']);R=B[:3,:3]@A[:3,:3].T;t=B[:3,3]-R@A[:3,3]
    x,y,z=t;cross=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    F=np.linalg.inv(second['K']).T@cross@R@np.linalg.inv(first['K'])
    norm=np.linalg.norm(F)
    return None if norm<1e-14 else F/norm


def pair_sum_bound(first,second):
    F=fundamental(first,second)
    if F is None:return 0.
    c1=np.asarray(first['kp_conf']);c2=np.asarray(second['kp_conf']);good=(c1>=.2)&(c2>=.2)
    if not good.any():return 0.
    x1=np.c_[np.asarray(first['kps'])[good],np.ones(sum(good))]
    x2=np.c_[np.asarray(second['kps'])[good],np.ones(sum(good))]
    w1=np.sqrt(c1[good]);w2=np.sqrt(c2[good]);Fx=(F@x1.T).T;Ftx=(F.T@x2.T).T
    e=abs(np.sum(x2*Fx,axis=1))
    a=np.maximum(np.max(abs(Ftx[:,:2]),axis=1)/w1,np.max(abs(Fx[:,:2]),axis=1)/w2)
    b=float(np.max(abs(F[:2,:2])))/(4*w1*w2)
    denominator=a+np.sqrt(a*a+4*b*e)
    lower=np.divide(2*e,denominator,out=np.zeros_like(e),where=denominator>1e-20)
    return float(lower.sum())


def mean_residual_lower_bound(picks):
    if len(picks)<2:return 0.
    count=sum(2*np.count_nonzero(np.asarray(p['kp_conf'])>=.2) for p in picks)
    if count==0:return 0.
    bounds=[pair_sum_bound(a,b) for a,b in itertools.combinations(picks,2)]
    return max(max(bounds,default=0.),sum(bounds)/(len(picks)-1))/count


def impossible_under_original_gate(picks):
    return mean_residual_lower_bound(picks)>5.+1e-8

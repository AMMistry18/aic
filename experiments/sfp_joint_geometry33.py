"""Experimental dual-port fit; public geometry, calibrated RGB keypoints only.

This is an offline candidate until prospective physical qualification. Detector
scores weight residuals; they are not a calibrated pose uncertainty estimate.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

CORNERS=np.array([[.00685,.0043,0],[-.00685,.0043,0],[-.00685,-.0043,0],[.00685,-.0043,0]])
SLOT1=np.array([-.0232,0.,0.])
POINTS=np.r_[CORNERS,CORNERS+SLOT1]


def triangulate(pixels,projections,weights):
    rows=[]
    for uv,P,weight in zip(pixels,projections,weights):
        rows.extend([np.sqrt(weight)*(uv[0]*P[2]-P[0]),np.sqrt(weight)*(uv[1]*P[2]-P[1])])
    _,_,vt=np.linalg.svd(rows);x=vt[-1]
    if abs(x[3])<1e-12:raise ValueError('degenerate triangulation')
    return x[:3]/x[3]


def fit(picks):
    if len(picks)<2:return dict(accepted=False,reason='insufficient_views')
    indices=[];tri=[];weights=[]
    for i in range(8):
        good=[p for p in picks if np.isfinite(p['kps'][i]).all() and p['kp_conf'][i]>=.2]
        if len(good)<2:continue
        tri.append(triangulate([p['kps'][i] for p in good],[p['P'] for p in good],[p['kp_conf'][i] for p in good]))
        indices.append(i);weights.append(sum(p['kp_conf'][i] for p in good))
    if len(indices)<4:return dict(accepted=False,reason='insufficient_triangulated_points')
    local=POINTS[indices];tri=np.asarray(tri);w=np.asarray(weights);w/=sum(w)
    a=np.sum(w[:,None]*local,axis=0);b=np.sum(w[:,None]*tri,axis=0)
    U,_,Vt=np.linalg.svd((w[:,None]*(local-a)).T@(tri-b));R=Vt.T@np.diag([1,1,np.linalg.det(Vt.T@U.T)])@U.T;p=b-R@a
    shape=float(np.sqrt(np.sum(w*np.sum(((R@local.T).T+p-tri)**2,axis=1))))
    if shape>.0035:return dict(accepted=False,reason='raw_shape_error',raw_shape_rmse_m=shape)
    def pose(x):return p+x[:3]*.001,Rotation.from_rotvec(np.deg2rad(x[3:])).as_matrix()@R
    def optimize(selected):
        def residual(x):
            position,rotation=pose(x);world=(rotation@POINTS.T).T+position;rows=[]
            for pick in selected:
                good=np.asarray(pick['kp_conf'])>=.2;uv=(pick['P']@np.c_[world,np.ones(8)].T).T
                if np.any(uv[:,2]<=.001):return np.full(2*sum(np.count_nonzero(np.asarray(q['kp_conf'])>=.2) for q in selected),1e4)
                error=uv[:,:2]/uv[:,2:]-pick['kps'];rows.extend((error[good]*np.sqrt(np.asarray(pick['kp_conf'])[good,None])).ravel())
            return np.array(rows)
        return least_squares(residual,np.zeros(6),bounds=([-4]*3+[-10]*3,[4]*3+[10]*3),loss='soft_l1',f_scale=1,diff_step=1e-4,max_nfev=120)
    best=optimize(picks);position,rotation=pose(best.x);per_view=[]
    for pick in picks:
        uv=(pick['P']@np.c_[(rotation@POINTS.T).T+position,np.ones(8)].T).T;good=np.asarray(pick['kp_conf'])>=.2
        per_view.append(float(np.mean(np.linalg.norm((uv[:,:2]/uv[:,2:]-pick['kps'])[good],axis=1))))
    loo=[]
    if len(picks)>=3:
        for omit in range(len(picks)):
            pair=optimize([q for i,q in enumerate(picks) if i!=omit]);pp,rr=pose(pair.x)
            loo.append(dict(translation_m=float(np.linalg.norm(pp-position)),angle_deg=float(np.rad2deg(Rotation.from_matrix(rotation.T@rr).magnitude()))))
    return dict(accepted=bool(best.success and max(per_view)<=5),reason='fitted' if best.success and max(per_view)<=5 else 'reprojection',
                position=position.tolist(),rotation=rotation.tolist(),raw_shape_rmse_m=shape,
                per_view_reprojection_px=per_view,leave_one_view_out=loo,
                GT_policy_inputs=False)

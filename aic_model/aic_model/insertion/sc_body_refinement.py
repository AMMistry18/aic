"""Image-only SC translation refinement conditional on aligned upstream handoff.

Uses a public CAD body hull, robot self-mask, camera calibration, image-time
robot pose and learned proposal. Port rotation is a declared angular prior.
No known connector grasp, object truth or simulator segmentation is consumed.
"""
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from .sc_rim_refinement import project


@lru_cache(maxsize=1)
def template():
    with np.load(Path(__file__).with_name('sc_body_template.npz')) as data:
        return data['body_hull_m'].copy(),data['tool_triangles_tcp_m'].copy()


def aligned_orientation_prior(port_rotation,measured_rotation,*,skip_consistency_for_ablation=False):
    port=np.asarray(port_rotation);measured=np.asarray(measured_rotation)
    candidates=[port,port@np.diag([-1.,-1.,1.])]
    angles=[float(np.arccos(np.clip((np.trace(R.T@measured)-1)*.5,-1,1))) for R in candidates]
    index=int(np.argmin(angles))
    # At 15.64 mm depth this discrepancy consumes the 0.25 mm lateral budget.
    # A necessary learned-orientation consistency screen, not certification of
    # the true handoff angle or of the estimator's uncertainty.
    limit=float(np.arctan2(.00025,.01564))
    diagnostic={'measured_prior_mismatch_deg':float(np.rad2deg(angles[index])),
                'prior_consistency_limit_deg':float(np.rad2deg(limit)),
                'port_symmetry_branch_deg':index*180,'angular_prior_certified':False,
                'angular_consistency_screen_applied':not skip_consistency_for_ablation,
                'diagnostic_ablation':bool(skip_consistency_for_ablation)}
    # Explicit research ablation of this estimator-based veto only. The caller
    # must already require the aligned-handoff condition. This does not certify
    # actual alignment, and its trials are not guarded-policy successes.
    return (candidates[index] if skip_consistency_for_ablation or angles[index]<=limit else None),diagnostic


def robot_tool_mask(K,T_camera_from_base,T_base_from_tcp,shape):
    """Rasterize robot-owned tool geometry with the current calibrated camera."""
    _,triangles=template();T=np.asarray(T_camera_from_base)@np.asarray(T_base_from_tcp)
    points=(T[:3,:3]@triangles.reshape(-1,3).T).T+T[:3,3]
    camera=points.reshape(-1,3,3);valid=np.all(camera[:,:,2]>.001,axis=1)
    camera=camera[valid];uv=(np.asarray(K)@camera.reshape(-1,3).T).T
    uv=(uv[:,:2]/uv[:,2:]-.5).reshape(-1,3,2)
    h,w=shape[:2];inside=(uv[:,:,0].max(1)>=0)&(uv[:,:,0].min(1)<w)&(uv[:,:,1].max(1)>=0)&(uv[:,:,1].min(1)<h)
    mask=np.zeros((h,w),np.uint8)
    for triangle in uv[inside]:cv2.fillConvexPoly(mask,np.rint(triangle).astype(np.int32),255)
    return mask


def observations(views,points,position,rotation):
    directions=np.c_[np.cos(np.linspace(0,2*np.pi,96,endpoint=False)),np.sin(np.linspace(0,2*np.pi,96,endpoint=False))]
    result=[]
    for view in views:
        image=view['image'];hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        uv=project(view['P'],(rotation@points.T).T+position)-.5
        if not np.isfinite(uv).all():raise ValueError('invalid_body_projection')
        ignored=cv2.dilate(view['tool_mask'],np.ones((3,3),np.uint8))!=0
        orange=cv2.inRange(hsv,np.array([5,100,60]),np.array([35,255,255]))!=0
        occluded=ignored|orange
        expanded=cv2.dilate(occluded.astype(np.uint8),np.ones((5,5),np.uint8))!=0
        roi=np.zeros(image.shape[:2],np.uint8)
        cv2.fillConvexPoly(roi,cv2.convexHull(np.rint(uv).astype(np.int32)),255)
        roi=cv2.dilate(roi,np.ones((15,15),np.uint8))!=0
        white=(hsv[:,:,1]<60)&(hsv[:,:,2]>80)&~occluded&roi
        y,x=np.nonzero(white)
        if len(x)<50:raise ValueError('insufficient_white_pixels')
        observed=np.c_[x,y].astype(float)
        model_ids=np.argmax(uv@directions.T,axis=0);obs_ids=np.argmax(observed@directions.T,axis=0)
        def visible(coords):
            xy=np.rint(coords).astype(int);h,w=image.shape[:2]
            inside=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
            valid=inside.copy();valid[inside]&=~expanded[xy[inside,1],xy[inside,0]]
            return valid
        keep=visible(uv[model_ids])&visible(observed[obs_ids]);D=directions[keep]
        if len(D)<12:raise ValueError('insufficient_unoccluded_directions')
        support=np.max(observed@D.T,axis=0);boundary=observed[obs_ids[keep]]
        s=np.linspace(-2.5,2.5,51);query=boundary[:,None,:]+D[:,None,:]*s[None,:,None]
        colors=np.stack([map_coordinates(image[:,:,c].astype(float),query[:,:,::-1].reshape(-1,2).T,order=1,mode='nearest').reshape(len(D),len(s)) for c in range(3)],axis=-1)
        foreground=np.median(colors[:,:10],axis=1);background=np.median(colors[:,-10:],axis=1)
        contrast=foreground-background;norm=np.sum(contrast**2,axis=1)
        alpha=np.sum((colors-background[:,None,:])*contrast[:,None,:],axis=2)/np.maximum(norm[:,None],1.)
        offsets=[];valid=[]
        for i in range(len(D)):
            transitions=np.flatnonzero((alpha[i,:-1]>=.5)&(alpha[i,1:]<.5))
            if norm[i]<30**2 or not len(transitions):continue
            j=min(transitions,key=lambda k:abs(s[k]))
            offset=s[j]+(.5-alpha[i,j])*(s[j+1]-s[j])/(alpha[i,j+1]-alpha[i,j])
            if abs(offset)>1.5:continue
            valid.append(i);offsets.append(offset)
        if len(valid)<12:raise ValueError('insufficient_subpixel_boundary_support')
        result.append((view['P'],D[valid],support[valid]+offsets))
    return result


def fit_translation(views,position,rotation):
    """Three-view body fit with 1 px residual and 0.25 mm LOO gates."""
    if len(views)!=3 or len({v['camera'] for v in views})!=3:
        return {'accepted':False,'reason':'three_unique_views_required'}
    position=np.asarray(position,dtype=float);rotation=np.asarray(rotation,dtype=float)
    if not np.isfinite(position).all() or not np.isfinite(rotation).all():
        return {'accepted':False,'reason':'invalid_pose'}
    points,_=template()
    try:prepared=observations(views,points,position,rotation)
    except ValueError as exc:return {'accepted':False,'reason':str(exc)}
    world=(rotation@points.T).T+position
    def optimize(selected):
        def residual(delta):
            return np.concatenate([np.max((project(P,world+delta*.001)-.5)@D.T,axis=0)-support for P,D,support in selected])
        fits=[least_squares(residual,[x,y,0],bounds=([-3]*3,[3]*3),loss='soft_l1',f_scale=1,diff_step=1e-4,max_nfev=150)
              for x,y in [(0,0),(-.5,0),(.5,0),(0,-.5),(0,.5)]]
        best=min(fits,key=lambda x:x.cost)
        return best,float(np.mean(np.abs(residual(best.x))))
    best,error=optimize(prepared);position_fit=position+best.x*.001
    disagreements=[];vectors=[]
    for omit in range(3):
        pair,_=optimize([row for i,row in enumerate(prepared) if i!=omit])
        disagreements.append(float(np.linalg.norm(pair.x-best.x)*.001))
        vectors.append(((pair.x-best.x)*.001).tolist())
    if not best.success:reason='optimizer_failed'
    elif np.linalg.norm(best.x)>3. or np.any(abs(best.x)>=2.99):reason='correction_bound'
    elif error>1.:reason='body_support_residual'
    elif max(disagreements)>.00025:reason='leave_one_view_out_disagreement'
    else:reason='accepted'
    return {'accepted':reason=='accepted','reason':reason,'position':position_fit.tolist(),
            'prior_rotation':rotation.tolist(),'correction_base_m':(best.x*.001).tolist(),
            'mean_support_error_px':error,'directions_per_view':[len(D) for _,D,_ in prepared],
            'leave_one_view_out_disagreement_m':disagreements,
            'leave_one_view_out_vectors_base_m':vectors,
            'partial_translation':partial_translation_contract(vectors,rotation,
                usable_fit=bool(best.success and np.linalg.norm(best.x)<=3. and np.all(abs(best.x)<2.99) and error<=1.)),
            'raster_offset_px':-.5,
            'orientation_assumption':'upstream_aligned_to_visual_port','simulator_object_inputs':False}


def partial_translation_contract(vectors,rotation,*,usable_fit):
    """Separate lateral image evidence from axial contact capture capability.

    The original 3D fit verdict remains unchanged. This additional contract can
    only be used by a policy that requires a subsequent independent axial
    contact measurement. It is not calibrated uncertainty or a success verdict.
    """
    delta=np.asarray(vectors,dtype=float)@np.asarray(rotation,dtype=float)
    valid=delta.shape==(3,3) and np.isfinite(delta).all()
    lateral=float(np.max(np.linalg.norm(delta[:,:2],axis=1))) if valid else float('inf')
    axial=float(np.max(abs(delta[:,2]))) if valid else float('inf')
    return dict(lateral_candidate_qualified=bool(usable_fit and lateral<=.00025 and axial<=.0005),
                maximum_lateral_disagreement_m=lateral,unqualified_axial_disagreement_m=axial,
                lateral_budget_m=.00025,axial_contact_capture_bound_m=.0005,
                requires_axial_contact_measurement=True,full_visual_pose_certified=False)

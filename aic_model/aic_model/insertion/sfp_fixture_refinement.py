"""Conditional SFP RGB measurement: learned proposals and public dual-rim CAD.

Level fixture family is an explicit workcell assumption. No simulator frames,
task target poses or prescribed grasp transform enter this component.
"""
import itertools
import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

CORNERS = np.array([[.00685,.0043,0],[-.00685,.0043,0],[-.00685,-.0043,0],[.00685,-.0043,0]])
SLOT1 = np.array([-.0232,0.,0.])
POINTS = np.r_[CORNERS,CORNERS+SLOT1]
FIXED = Rotation.from_euler('x',3.12895).as_matrix()


def project(P, world):
    q=(P@np.c_[world,np.ones(len(world))].T).T
    if np.any(q[:,2]<=0):raise ValueError('CAD behind camera')
    return q[:,:2]/q[:,2:]


def fit_fixture(picks):
    first=picks[0]
    _,rs,ts,_=cv2.solvePnPGeneric(POINTS,first['kps'].astype(float),first['K'],None,flags=cv2.SOLVEPNP_IPPE)
    fits=[]
    for r,t in zip(rs,ts):
        camera=np.eye(4);camera[:3,:3]=cv2.Rodrigues(r)[0];camera[:3,3]=t.ravel()
        world=np.linalg.inv(first['T'])@camera
        if world[2,2]>-.8:continue
        origin=world[:3,3];yaw=np.rad2deg(np.arctan2(world[1,0],world[0,0]))
        def pose(x):return origin+x[:3]*.001,Rotation.from_euler('z',yaw+x[3],degrees=True).as_matrix()@FIXED
        def residual(x):
            p,R=pose(x);points=(R@POINTS.T).T+p;rows=[]
            for pick in picks:
                good=np.asarray(pick['kp_conf'])>=.2
                rows.extend(((project(pick['P'],points)-pick['kps'])[good]*np.sqrt(np.asarray(pick['kp_conf'])[good,None])).ravel())
            return np.array(rows)
        best=least_squares(residual,np.zeros(4),bounds=([-20]*4,[20]*4),loss='soft_l1',f_scale=1,diff_step=1e-4,max_nfev=150)
        p,R=pose(best.x);error=float(np.mean(abs(residual(best.x))))
        fits.append((best,error,p,R))
    if not fits:return None
    best,error,p,R=min(fits,key=lambda f:f[0].cost)
    if not best.success or error>5:return None
    return dict(position=p.tolist(),rotation=R.tolist(),residual_px=error)


def fit_plug_translation(views,detections,initial,R,local):
    data={v.camera_name:v for v in views}
    if len(detections)<2:raise ValueError('Plug translation needs two cameras')
    def residual(x):
        world=(R@local.T).T+initial+x*.001;rows=[]
        for d in detections:
            view=data[d.camera_name];P=view.K@np.linalg.inv(view.T_world_from_camera)[:3]
            good=np.asarray(d.keypoint_confidences)>=.15
            rows.extend(((project(P,world)-d.keypoints_px)[good]*np.sqrt(d.keypoint_confidences[good,None])).ravel())
        return np.array(rows)
    best=least_squares(residual,np.zeros(3),bounds=([-5]*3,[5]*3),loss='soft_l1',f_scale=1,diff_step=1e-4,max_nfev=150)
    error=float(np.mean(abs(residual(best.x))))
    if not best.success or error>6 or max(abs(best.x))>=4.99:raise ValueError('Plug translation fit unqualified')
    return initial+best.x*.001,error


def select_candidate(candidates,plug):
    """Last-inch association needs stereo support, lateral proximity and gap.

    A single-view small Euclidean distance can be a depth ghost. Maximize
    agreeing view support before ranking lateral proximity to the held plug.
    Exact reset gap and target position are never supplied.
    """
    eligible=[]
    for c in candidates:
        relative=np.array(c['rotation']).T@(plug-np.array(c['position']))
        if len(c['cameras'])>=2 and -.025<=relative[2]<=-.003 and np.linalg.norm(relative[:2])<=.004:
            eligible.append((c,float(np.linalg.norm(relative[:2]))))
    if not eligible:raise ValueError('No stereo NIC candidate in last-inch capture envelope')
    return min(eligible,key=lambda row:(-len(row[0]['cameras']),row[1]))[0]


def refine_rim(views,seed,plug):
    from .sfp_plug_pose_geometry import SFP_PLUG_LOCAL_KEYPOINTS_M
    p0=np.array(seed['position']);R0=np.array(seed['rotation']);prepared=[]
    # All eight public rim edges; the hidden plug footprint is excluded.
    edges=[np.linspace(c[j],c[(j+1)%4],35) for c in (CORNERS,CORNERS+SLOT1) for j in range(4)]
    if seed['slot']==1:edges=[e-SLOT1 for e in edges]
    for v in views:
        P=v.K@np.linalg.inv(v.T_world_from_camera)[:3]
        gray=cv2.cvtColor(v.image_bgr,cv2.COLOR_BGR2GRAY)
        all_edges=cv2.Canny(gray,12,35);gx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3);gy=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
        magnitude=np.hypot(gx,gy);mask=np.zeros(gray.shape,np.uint8)
        uv=project(P,(R0@SFP_PLUG_LOCAL_KEYPOINTS_M.T).T+plug)-.5
        cv2.fillConvexPoly(mask,cv2.convexHull(np.rint(uv).astype(np.int32)),255)
        mask=cv2.dilate(mask,np.ones((9,9),np.uint8));fields=[]
        for points in edges:
            uv=project(P,(R0@points.T).T+p0)-.5;direction=uv[-1]-uv[0]
            normal=np.array([direction[1],-direction[0]])/np.linalg.norm(direction)
            aligned=abs(gx*normal[0]+gy*normal[1])>=magnitude*np.cos(np.deg2rad(20))
            edge=all_edges.copy();edge[~aligned]=0;edge[mask!=0]=0
            fields.append(cv2.distanceTransform(255-edge,cv2.DIST_L2,cv2.DIST_MASK_PRECISE).astype(float))
        prepared.append((P,mask,fields))
    def pose(x):return p0+x[:3]*.001,Rotation.from_euler('z',x[3],degrees=True).as_matrix()@R0
    def residual(x,counts=False):
        p,R=pose(x);rows=[];visibility=[]
        for P,mask,fields in prepared:
            visible_count=0
            for points,field in zip(edges,fields):
                uv=project(P,(R@points.T).T+p)-.5
                visible=map_coordinates(mask,uv[:,::-1].T,order=0,mode='constant',cval=255)==0
                visible_count+=int(sum(visible))
                values=map_coordinates(field,uv[:,::-1].T,order=1,mode='constant',cval=20)
                rows.extend(np.where(visible,np.minimum(values,12),0.))
            visibility.append(visible_count)
        return visibility if counts else np.array(rows)
    options=[least_squares(residual,[x,y,z,0.],bounds=([-4,-4,-4,-5],[4,4,4,5]),loss='soft_l1',f_scale=1,diff_step=1e-4,max_nfev=100)
             for x,y,z in itertools.product([0.,-1.5,1.5],repeat=3)]
    # Reject fits which hide almost the entire geometry in the plug mask.
    valid=[o for o in options if o.success and min(residual(o.x,True))>=70 and max(abs(o.x[:3]))<3.99]
    if not valid:raise ValueError('Rim fit insufficient visibility or outside capture range')
    best=min(valid,key=lambda x:x.cost);error=float(np.mean(residual(best.x)))
    if error>2:raise ValueError(f'Rim image residual {error:.3f}px exceeds 2px')
    p,R=pose(best.x)
    return p,R,dict(residual_px=error,visible_samples_per_view=residual(best.x,True),correction_mm_deg=best.x.tolist(),precision_certified=False)


def perceive(policy,observation):
    from .sfp_controller import _plug_views_from_observation,solve_tip_in_tcp
    from .sfp_plug_pose_geometry import SFP_PLUG_LOCAL_KEYPOINTS_M
    views=_plug_views_from_observation(policy,observation)
    # Check age at acquisition, as the original priming path does. Detector
    # compute time is not a camera transport delay. The estimate is anchored
    # to exposure-time TCP below, then robot stationarity is checked before
    # motion; no old exposure is relabelled with a newer stamp.
    now=policy._parent_node.get_clock().now().nanoseconds*1e-9
    # Retain actual exposure-time robot poses before expensive inference/CAD
    # fitting can age them out of TF history. No current/latest-TF substitute.
    exposure_tcp={}
    for stamp in sorted({float(v.stamp_s) for v in views}):
        try:position,orientation=policy._tcp_at_stamp(stamp)
        except Exception as exc:
            raise ValueError(f'Exposure TCP unavailable at {stamp:.9f}s: {exc}') from exc
        exposure_tcp[stamp]=(np.asarray(position,dtype=float).copy(),np.asarray(orientation,dtype=float).copy())
    estimator=policy._sfp_plug_estimator;detections=estimator.detect_views(views)
    measured=estimator.estimate_multiview(views,now_s=now,max_age_s=policy._sfp_config.plug_max_age_s,detections=detections)
    if measured is None:raise ValueError('Fresh original plug gate: '+str(estimator.last_failure_reason))
    cams={}
    for v in views:
        T=np.linalg.inv(v.T_world_from_camera)
        ds=policy._pc.detect_nic(v.image_bgr,conf_thresh=.2)
        if ds:cams[v.camera_name]=[dict(kps=d['kps'],kp_conf=d['kp_conf'],K=v.K,T=T,P=v.K@T[:3]) for d in ds[:5]]
    candidates=[]
    for count in range(2,len(cams)+1):
        for subset in itertools.combinations(cams,count):
            for picks in itertools.product(*(cams[c] for c in subset)):
                fit=fit_fixture(picks)
                if fit is None:continue
                for slot,offset in enumerate((np.zeros(3),SLOT1)):
                    p=np.array(fit['position'])+np.array(fit['rotation'])@offset
                    candidates.append(dict(fit,position=p.tolist(),slot=slot,cameras=list(subset)))
    seed=select_candidate(candidates,measured.position_world)
    plug,plug_error=fit_plug_translation(views,detections,measured.position_world,np.array(seed['rotation']),SFP_PLUG_LOCAL_KEYPOINTS_M)
    p,R,rim=refine_rim(views,seed,plug)
    # Translation is visually measured; retain measured G rotation so holding
    # the initial estimated tip rotation preserves the initial physical wrist.
    if float(measured.stamp_s) not in exposure_tcp:
        raise ValueError('Visual estimate stamp has no captured exposure TCP')
    tcp,quat=exposure_tcp[float(measured.stamp_s)]
    current_tcp,current_quat=policy._tcp()
    from .contract import quat_to_matrix
    motion_m=float(np.linalg.norm(np.asarray(current_tcp)-tcp))
    motion_deg=float(np.rad2deg(Rotation.from_matrix(quat_to_matrix(current_quat)@quat_to_matrix(quat).T).magnitude()))
    if motion_m>.0001 or motion_deg>.05:raise ValueError('Wrist moved during stationary SFP fixture measurement')
    policy._sfp_grasp_transform=solve_tip_in_tcp(tcp,quat,plug,measured.rotation_world_from_plug)
    policy._sfp_fixture_diagnostic=dict(seed=seed,rim=rim,plug_translation_residual_px=plug_error,
        exposure_tcp_retained_before_inference=True,exposure_tcp_stamp_s=float(measured.stamp_s),
        grasp_translation_from_RGB=True,assumption='level NIC source family and upstream angular alignment',
        camera_stamps_s=[v.stamp_s for v in views],input_age_s=measured.age_s,
        processing_sim_s=policy._parent_node.get_clock().now().nanoseconds*1e-9-now,
        exposure_to_current_tcp_motion_m=motion_m,exposure_to_current_tcp_motion_deg=motion_deg,
        ground_truth_used=False,precision_certified=False)
    return p,R

"""Image-only view-subset/semantic-occlusion diagnostic, all variants reported.

No subset is chosen by simulator error or angular-prior agreement. The adaptive
variant rejects a view only from observed orange overlap with its public CAD
body projection predicted using the OTHER two views.
"""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'aic_model')]
from aic_model.insertion.sfp_plug_pose import PlugPoseView,PlugKeypointDetection,fuse_multiview_keypoints
from aic_model.insertion.sc_plug_pose_geometry import SC_PLUG_LOCAL_KEYPOINTS_M
from aic_model.insertion.sc_body_refinement import template,robot_tool_mask
from aic_model.insertion.sc_rim_refinement import project
from experiments.lastinch_geometry import HOME,robot_fk


def main():
    source=ROOT/'.artifacts/reference30/foreground_anchor30_last/fits_without_truth.json';reference=json.loads(source.read_text())['rows']
    out=ROOT/'.artifacts/reference30/foreground_view_subsets30'
    if out.exists():raise FileExistsError('Preserve prior diagnostic')
    out.mkdir(parents=True);data=ROOT/'.artifacts/perception_data29/qualified';fk=robot_fk(ET.parse(ROOT/'.artifacts/lastinch_resources/ur5e.urdf').getroot(),HOME);C={c:np.linalg.inv(fk[c+'/optical'])@fk['gripper/tcp'] for c in ['left_camera','center_camera','right_camera']};points,_=template();rows=[]
    for r in reference:
        views=[]
        if 'case_id' in r:
            name=r['case_id'];capture=json.loads((data/'cases'/name/'training_capture.json').read_text())
            for c,v in capture['cameras'].items():views.append(PlugPoseView(c,cv2.imread(str(data/'dataset/sc_plug/images/val'/(name+'_'+c+'.png'))),np.array(v['K']),np.array(v['T_world_from_camera']),v['image_stamp_s'],c))
        else:
            root=ROOT/'.artifacts/lastinch_runs'/r['run']/r['run']/r['capture'];geometry=json.loads((root/'camera_geometry.json').read_text())
            for v in geometry:
                c=v['camera'];views.append(PlugPoseView(c,cv2.imread(str(root/(c+'_raw.png'))),np.array(v['K']),np.array(v['T_base_from_camera']),v['stamp_s'],c))
        detections=[PlugKeypointDetection(d['camera'],np.array(d['keypoints_px']),np.array(d['confidence']),1.) for d in r['detections']]
        subsets=dict(three=list(C),left_right=['left_camera','right_camera'],left_center=['left_camera','center_camera'],center_right=['center_camera','right_camera']);fits={}
        def fit(names):
            vs=[v for v in views if v.camera_name in names];ds=[d for d in detections if d.camera_name in names]
            p,R,shape,error,count,conf=fuse_multiview_keypoints(vs,ds,local_keypoints_m=SC_PLUG_LOCAL_KEYPOINTS_M)
            return p,R,dict(shape_mm=shape*1000,reprojection_px=error,original_geometry_gates_passed=shape<=.0035 and error<=6)
        for mode,names in subsets.items():
            try:fits[mode]=fit(names)
            except ValueError:pass
        occlusion={}
        for v in views:
            pair=[c for c in C if c!=v.camera_name]
            try:
                pos,R,quality=fit(pair);uv=project(v.K@np.linalg.inv(v.T_world_from_camera)[:3],(R@points.T).T+pos)-.5
                roi=np.zeros(v.image_bgr.shape[:2],np.uint8);cv2.fillConvexPoly(roi,cv2.convexHull(np.rint(uv).astype(np.int32)),255)
                robot=cv2.dilate(robot_tool_mask(v.K,np.eye(4),C[v.camera_name],v.image_bgr.shape),np.ones((3,3),np.uint8))!=0;visible=(roi!=0)&~robot
                orange=cv2.inRange(cv2.cvtColor(v.image_bgr,cv2.COLOR_BGR2HSV),np.array([5,100,60]),np.array([35,255,255]))!=0
                area=int(visible.sum());overlap=int(np.count_nonzero(visible&orange));fraction=overlap/max(area,1)
                occlusion[v.camera_name]=dict(projected_visible_area_px=area,orange_overlap_px=overlap,occlusion_fraction=fraction,other_two_views_geometry_passed=quality['original_geometry_gates_passed'],accepted=quality['original_geometry_gates_passed'] and area>=50 and fraction<=.05)
            except ValueError as e:occlusion[v.camera_name]=dict(accepted=False,reason=str(e))
        selected=[c for c in C if occlusion[c]['accepted']]
        if len(selected)>=2:
            try:fits['adaptive_orange']=fit(selected)
            except ValueError:pass
        for mode in [*subsets,'adaptive_orange']:
            row={k:r[k] for k in ['case_id','run','capture','split','camera_stamps_s'] if k in r};row.update(mode=mode,semantic_occlusion=occlusion,selected_cameras=selected if mode=='adaptive_orange' else subsets[mode],simulator_truth_for_selection=False)
            if mode in fits:
                pos,R,quality=fits[mode];pose=np.eye(4);pose[:3,:3]=R;pose[:3,3]=pos;own_tcp=views[0].T_world_from_camera@C[views[0].camera_name]
                row.update(position=pos.tolist(),rotation=R.tolist(),estimated_grasp_tcp=(np.linalg.inv(own_tcp)@pose).tolist(),**quality)
            else:row['failure']='insufficient_or_inconsistent_views'
            rows.append(row)
    (out/'fits_without_truth.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),orange_occlusion_fraction_limit=.05,selection_uses_other_views_only=True,rows=rows),indent=2)+'\n')
    invariance=[]
    for mode in [*subsets,'adaptive_orange']:
        for name in dict.fromkeys(r['run'] for r in rows if r.get('run')):
            pair=[r for r in rows if r.get('run')==name and r['mode']==mode]
            if len(pair)!=2:continue
            if any('estimated_grasp_tcp' not in r for r in pair):invariance.append(dict(mode=mode,run=name,passed=False));continue
            g0,g1=[np.array(r['estimated_grasp_tcp']) for r in pair];dt=float(np.linalg.norm(g0[:3,3]-g1[:3,3])*1000);dr=float(np.rad2deg(Rotation.from_matrix(g0[:3,:3].T@g1[:3,:3]).magnitude()))
            invariance.append(dict(mode=mode,run=name,translation_change_mm=dt,rotation_change_deg=dr,passed=dt<=.25 and dr<=.25))
    (out/'image_only_invariance.json').write_text(json.dumps(invariance,indent=2)+'\n')
    B=np.eye(4);B[:3,:3]=Rotation.from_euler('z',-np.pi).as_matrix();B[:3,3]=[-.2,.2,0]
    for r in rows:
        if 'position' not in r:continue
        if 'case_id' in r:truth=np.array(json.loads((data/'cases'/r['case_id']/'scene.json').read_text())['reset']['tip_world'])
        else:
            trace=json.load(gzip.open(ROOT/'.artifacts/lastinch_runs'/r['run']/r['run']/'verification_trace.json.gz','rt'));near=min(trace,key=lambda x:abs(x['sim_time_s']-max(r['camera_stamps_s'])));truth=np.linalg.inv(B)@near['tip_world']
        r.update(position_error_mm=float(np.linalg.norm(np.array(r['position'])-truth[:3,3])*1000),angle_error_deg=float(np.rad2deg(Rotation.from_matrix(truth[:3,:3].T@r['rotation']).magnitude())))
    (out/'diagnostic_scores.json').write_text(json.dumps(rows,indent=2)+'\n');summary={}
    for mode in [*subsets,'adaptive_orange']:
        static=[r for r in rows if r['mode']==mode and r['split']=='static_development'];live=[r for r in rows if r['mode']==mode and r['split']=='recorded_live_development']
        summary[mode]=dict(static_median_angle_deg=float(np.median([r.get('angle_error_deg',180) for r in static])),static_max_angle_deg=max(r.get('angle_error_deg',180) for r in static),live_median_angle_deg=float(np.median([r.get('angle_error_deg',180) for r in live])),static_gated=sum(r.get('original_geometry_gates_passed',False) for r in static),invariance_passed=sum(r['passed'] for r in invariance if r['mode']==mode))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

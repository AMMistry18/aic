"""Offline physical camera and keypoint-contract audit of synchronized capture."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'aic_model'))
from aic_model.insertion.sc_plug_pose_geometry import SC_PLUG_LOCAL_KEYPOINTS_M as POINTS
from aic_model.insertion.sfp_plug_pose import triangulate_dlt

def matrix(t):
    t=t['transform'];M=np.eye(4)
    M[:3,3]=[t['translation'][k] for k in ('x','y','z')]
    M[:3,:3]=Rotation.from_quat([t['rotation'][k] for k in ('x','y','z','w')]).as_matrix()
    return M

def main():
    folder=ROOT/'.artifacts/kilted_handoff/projection_audit'
    audit=json.loads((folder/'audit.json').read_text());rows=audit['views'];metrics=[]
    base=np.eye(4);base[:3,:3]=Rotation.from_euler('z',-np.pi).as_matrix();base[:3,3]=[-.2,.2,0]
    for r in rows:
        frames={t['child_frame_id']:matrix(t) for t in r['robot_pose']['transforms']}
        actual=frames[f"ur5e/{r['camera']}/optical"]
        # PosePublisher robot links are in world coordinates (robot has no model pose).
        tf_world=base@np.linalg.inv(r['T_camera_from_base_exact'])
        delta=np.linalg.inv(actual)@tf_world
        latest=np.linalg.inv(r['T_base_from_camera_latest'])
        dlatest=np.linalg.inv(r['T_camera_from_base_exact'])@latest
        metrics.append({'camera':r['camera'],'extrinsic_translation_mm':float(np.linalg.norm(delta[:3,3])*1000),
            'extrinsic_rotation_deg':float(np.linalg.norm(Rotation.from_matrix(delta[:3,:3]).as_rotvec())*180/np.pi),
            'latest_vs_image_translation_mm':float(np.linalg.norm(dlatest[:3,3])*1000),
            'fx_from_hfov':1152/(2*np.tan(.8718/2)),'reported_fx':r['K'][0][0]})
    ds=[r for r in rows if 'predicted_px' in r];points=[]
    for i in range(8):
        ps=[np.array(r['K'])@np.array(r['T_camera_from_base_exact'])[:3] for r in ds]
        X=triangulate_dlt([np.array(r['predicted_px'])[i] for r in ds],ps,[r['confidence'][i] for r in ds])
        points.append((np.linalg.inv(ds[0]['tip_world'])@base@np.r_[X,1])[:3])
    points=np.array(points)
    result={'camera_metrics':metrics,'predicted_points_in_actual_tip_mm':(points*1000).tolist(),
        'source_points_in_tip_mm':(POINTS*1000).tolist(),
        'point_error_mm':(np.linalg.norm(points-POINTS,axis=1)*1000).tolist(),
        'predicted_edge_lengths_mm':[(np.linalg.norm(points[a]-points[b])*1000) for a,b in ((0,1),(1,2),(0,4))]}
    (folder/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()

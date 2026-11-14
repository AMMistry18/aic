"""Qualitative stationary image/GT projection audit; not a synchronized eval set."""
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'aic_model'))
from aic_model.insertion.sc_plug_pose_geometry import SC_PLUG_LOCAL_KEYPOINTS_M


def main():
    run=ROOT/'.artifacts/kilted_handoff_v1_before_ipc'
    transforms=json.loads((run/'cable_transforms.json').read_text())[0]['transforms']
    frames={}
    for row in transforms:
        transform=row['transform'];matrix=np.eye(4)
        matrix[:3,3]=[transform['translation'][k] for k in ('x','y','z')]
        matrix[:3,:3]=Rotation.from_quat([transform['rotation'][k] for k in ('x','y','z','w')]).as_matrix()
        frames[row['child_frame_id']]=matrix
    tip=frames['cable_2']@frames['cable_2/sc_tip_link']
    base=np.eye(4);base[:3,:3]=Rotation.from_euler('z',-np.pi).as_matrix();base[:3,3]=[-.2,.2,0]
    cameras=json.loads((run/'policy_images/camera_geometry.json').read_text())
    detections=json.loads((run/'live_policy_probe_primed.json').read_text())['plug_detections']
    rows=[];fig,axes=plt.subplots(1,len(detections),figsize=(10,5))
    for ax,det in zip(np.atleast_1d(axes),detections):
        camera=next(c for c in cameras if c['name']==det['camera'])
        T=np.linalg.inv(camera['T_base_from_camera'])@np.linalg.inv(base)@tip
        xyz=(T@np.column_stack([SC_PLUG_LOCAL_KEYPOINTS_M,np.ones(8)]).T)[:3]
        projected=np.array(camera['K'])@xyz;uv=(projected[:2]/projected[2]).T
        pred=np.array(det['keypoints_px'])
        image=cv2.cvtColor(cv2.imread(str(run/'policy_images'/f"{det['camera']}_raw.png")),cv2.COLOR_BGR2RGB)
        ax.imshow(image);ax.scatter(*uv.T,s=30,marker='+',c='lime',label='Source geometry projection')
        ax.scatter(*pred.T,s=15,marker='o',facecolors='none',edgecolors='red',label='Predicted keypoint')
        for i,(a,b) in enumerate(zip(uv,pred)):
            ax.plot([a[0],b[0]],[a[1],b[1]],color='yellow',linewidth=.6)
            ax.text(a[0],a[1],str(i),color='lime',fontsize=7)
        both=np.vstack([uv,pred]);lo=both.min(0)-30;hi=both.max(0)+30
        ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1]);ax.set_title(det['camera']);ax.legend(fontsize=7)
        rows.append({'camera':det['camera'],'projected_px':uv.tolist(),
                     'predicted_minus_projection_px':(pred-uv).tolist(),
                     'mean_pixel_distance':float(np.linalg.norm(pred-uv,axis=1).mean())})
    fig.suptitle('Stationary diagnostic: image and pose stamps differ; qualitative audit')
    fig.tight_layout();fig.savefig(ROOT/'.artifacts/container/projected_keypoints.png',dpi=150);plt.close(fig)
    out={'purpose':'qualitative_stationary_projection_audit_not_synchronized_accuracy',
         'rows':rows,'image_stamps':[c['stamp_s'] for c in cameras],
         'pose_stamp':transforms[0]['header']['stamp']}
    (ROOT/'.artifacts/container/projected_keypoints.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in row.items() if k in ('camera','mean_pixel_distance')} for row in rows]))


if __name__=='__main__':main()

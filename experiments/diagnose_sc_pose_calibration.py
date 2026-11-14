"""Train-only affine pose-calibration negative control, never deployed."""
import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
from calibrate_sc_keypoints30 import estimate
from experiments.lastinch_geometry import robot_fk,HOME


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    data=ROOT/'.artifacts/perception_data29/qualified';rows=json.loads((ROOT/'.artifacts/reference30/keypoint_calibration30/predictions_without_truth.json').read_text());robot=ET.parse(ROOT/'.artifacts/lastinch_resources/ur5e.urdf').getroot();measurements=[]
    for row in rows:
        folder=data/'cases'/row['case_id'];scene=json.loads((folder/'scene.json').read_text());tcp=robot_fk(robot,scene['robot_q'])['gripper/tcp'];pred=estimate(row['views'],{},'original');T=np.eye(4);T[:3,:3]=pred['rotation'];T[:3,3]=pred['position'];G=np.linalg.inv(tcp)@T
        m=dict(case_id=row['case_id'],split=row['split'],tcp=tcp,G=G)
        if row['split']=='train':m['label']=np.linalg.inv(tcp)@scene['reset']['tip_world']
        measurements.append(m)
    ref=measurements[0]['G'][:3,:3]
    def vector(T):return np.r_[T[:3,3]*1000,np.rad2deg(Rotation.from_matrix(T[:3,:3]@ref.T).as_rotvec())]
    train=[m for m in measurements if m['split']=='train'];X=np.array([vector(m['G']) for m in train]);Y=np.array([vector(m['label']) for m in train]);mean=X.mean(0);scale=X.std(0,ddof=1)
    def fit(x,y,alpha):
        A=np.c_[(x-mean)/scale,np.ones(len(x))];reg=np.diag([alpha]*6+[0]);return np.linalg.solve(A.T@A+reg,A.T@y)
    errors={}
    for alpha in [.01,.1,1,10,100]:
        predictions=[]
        for i in range(len(train)):
            keep=np.arange(len(train))!=i;coef=fit(X[keep],Y[keep],alpha);predictions.append(np.r_[(X[i]-mean)/scale,1]@coef)
        d=np.array(predictions)-Y;errors[alpha]=float(np.mean(np.linalg.norm(d[:,3:],axis=1)))
    alpha=min(errors,key=errors.get);coef=fit(X,Y,alpha);jacobian=(coef[:6].T/scale)[:,3:];svd=np.linalg.svd(jacobian[3:],compute_uv=False)
    model=dict(training_scenes=len(train),selection='leave-one-training-scene-out angular error only',alpha=alpha,training_cv_angle_error_deg=errors,
        input_mean=mean.tolist(),input_scale=scale.tolist(),coefficient=coef.tolist(),reference_rotation=ref.tolist(),angular_jacobian_singular_values=svd.tolist(),production_enabled=False)
    (a.out/'model_train_only.json').write_text(json.dumps(model,indent=2)+'\n');results=[]
    for m in measurements:
        if m['split']!='val':continue
        x=vector(m['G']);v=np.r_[(x-mean)/scale,1]@coef;G=np.eye(4);G[:3,3]=v[:3]*.001;G[:3,:3]=Rotation.from_rotvec(np.deg2rad(v[3:])).as_matrix()@ref
        results.append(dict(case_id=m['case_id'],G=G.tolist(),original_G=m['G'].tolist(),tcp=m['tcp'].tolist()))
    (a.out/'dev_fits_without_truth.json').write_text(json.dumps(results,indent=2)+'\n')
    for r in results:
        truth=np.linalg.inv(r['tcp'])@np.array(json.loads((data/'cases'/r['case_id']/'scene.json').read_text())['reset']['tip_world'])
        for label,key in [('original','original_G'),('calibrated','G')]:
            G=np.array(r[key]);r[label+'_angle_error_deg']=float(np.rad2deg(Rotation.from_matrix(truth[:3,:3].T@G[:3,:3]).magnitude()));r[label+'_position_error_mm']=float(np.linalg.norm(truth[:3,3]-G[:3,3])*1000)
    (a.out/'dev_diagnostic_scores.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(dict(alpha=alpha,training_cv=errors,angular_jacobian_singular_values=svd.tolist(),development=[{k:v for k,v in r.items() if k not in ['G','original_G','tcp']} for r in results]),indent=2))


if __name__=='__main__':main()

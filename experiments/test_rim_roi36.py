"""Check the clipped-field identity throughout sampled existing fit bounds."""
import importlib.util
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'.artifacts/repair36/rim_roi_candidate/sfp_fixture_roi.py'
spec=importlib.util.spec_from_file_location('roi_candidate',path);candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)


def test_cropped_field_retains_every_subclip_distance_in_fit_domain():
    rng=np.random.default_rng(6036);height,width=960,1280
    K=np.array([[900.,0,640.],[0,900.,480.],[0,0,1.]])
    P=K@np.c_[np.eye(3),np.zeros(3)]
    local=np.vstack([candidate.CORNERS,candidate.CORNERS+candidate.SLOT1])
    p=np.array([.02,-.01,.45]);nominal=local+p
    roi=candidate.distance_field_roi(P,nominal,np.linalg.norm(local,axis=1).max(),(height,width))
    x0,y0,x1,y1=roi
    assert (x1-x0)*(y1-y0)<height*width/4
    for density in [.002,.03]:
        binary=np.where(rng.random((height,width))<density,0,255).astype(np.uint8)
        full=cv2.distanceTransform(binary,cv2.DIST_L2,cv2.DIST_MASK_PRECISE).astype(float)
        cropped=cv2.distanceTransform(binary[y0:y1,x0:x1],cv2.DIST_L2,cv2.DIST_MASK_PRECISE).astype(float)
        for _ in range(150):
            xyz=rng.uniform(-.004,.004,3);yaw=rng.uniform(-5,5)
            world=(Rotation.from_euler('z',yaw,degrees=True).as_matrix()@local.T).T+p+xyz
            uv=candidate.project(P,world)-.5
            expected=np.minimum(map_coordinates(full,uv[:,::-1].T,order=1,mode='constant',cval=20),12)
            observed=np.minimum(map_coordinates(cropped,(uv-[x0,y0])[:,::-1].T,order=1,mode='constant',cval=20),12)
            np.testing.assert_allclose(observed,expected,atol=1e-5,rtol=0)

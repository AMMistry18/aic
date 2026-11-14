"""Development-only multiview front-rim fit. No truth enters the fit."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
ROOT = Path(__file__).resolve().parents[1]


def project(P, points):
    q = (P @ np.c_[points, np.ones(len(points))].T).T
    return q[:, :2]/q[:, 2:]


def edge_distance(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, np.array([90,80,60]), np.array([130,255,255]))
    edge = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50, 100)
    edge &= cv2.dilate(blue, np.ones((3,3), np.uint8))
    occluded = ((hsv[:,:,1] < 60) & (hsv[:,:,2] > 80)).astype(np.uint8)*255
    occluded = cv2.dilate(occluded, np.ones((5,5), np.uint8))
    edge[occluded != 0] = 0
    return cv2.distanceTransform(255-edge, cv2.DIST_L2, cv2.DIST_MASK_PRECISE).astype(float), occluded


def fit_rim(views, position, rotation, *, raster_offset_px=0., full_rotation=False):
    points = np.load(ROOT/'experiments/configs/sc_front_rim.npz')['points_m']
    prepared = []
    for view in views:
        distance, occluded = edge_distance(view['image'])
        initial_uv = project(view['P'], (rotation @ points.T).T+position)+raster_offset_px
        visible = map_coordinates(occluded, initial_uv[:, ::-1].T, order=0, mode='constant', cval=255) == 0
        prepared.append((view, distance, visible))
    def pose(params):
        rot = np.asarray(params[3:]) if full_rotation else np.array([0,0,params[3]])
        return position+params[:3]*.001, Rotation.from_rotvec(rot*np.pi/180).as_matrix() @ rotation
    def residual(params):
        pos, R = pose(params); world=(R @ points.T).T+pos
        return np.concatenate([np.minimum(map_coordinates(field, (project(view['P'], world[visible])+raster_offset_px)[:, ::-1].T,
                                                          order=1, mode='constant', cval=20), 12)
                               for view, field, visible in prepared])
    fits = []
    for x,y in [(0,0),(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
        angles=3 if full_rotation else 1
        fit=least_squares(residual, [x,y,0]+[0]*angles, bounds=([-4,-4,-4]+[-10]*angles,[4,4,4]+[10]*angles),
                          loss='soft_l1', f_scale=1, max_nfev=100, diff_step=1e-4)
        fits.append(fit)
    best=min(fits, key=lambda fit: fit.cost)
    pos,R=pose(best.x); r=residual(best.x)
    return {'position': pos.tolist(), 'rotation': R.tolist(), 'delta_mm_deg': best.x.tolist(),
            'mean_edge_distance_px': float(np.mean(r)), 'median_edge_distance_px': float(np.median(r)),
            'p90_edge_distance_px': float(np.percentile(r,90)), 'cost': float(best.cost),
            'visible_points_per_view': [int(np.sum(v)) for _,_,v in prepared],
            'optimizer_success': bool(best.success), 'production_enabled': False, 'raster_offset_px':raster_offset_px,
            'full_rotation':full_rotation}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference-audit',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    audit=json.loads(a.reference_audit.read_text());folder=Path(audit['capture_folder'])
    T=np.array(audit['learned_pose_base']); geometry=json.loads((folder/'camera_geometry.json').read_text())
    views=[{'camera':v['camera'],'image':cv2.imread(str(folder/f'{v["camera"]}_raw.png')),
            'P':np.array(v['K']) @ np.linalg.inv(v['T_base_from_camera'])[:3]} for v in geometry]
    fit=fit_rim(views,T[:3,3],T[:3,:3]);(a.out/'fit_without_truth.json').write_text(json.dumps(fit,indent=2)+'\n')
    scene=json.loads((folder.parent.parent/'scene.json').read_text())
    base=np.eye(4);base[:3,:3]=Rotation.from_euler('z',-np.pi).as_matrix();base[:3,3]=[-.2,.2,0]
    truth=np.linalg.inv(base)@np.array(scene['reset']['mouth_world'])
    fit['diagnostic_before_error_mm']=float(np.linalg.norm(T[:3,3]-truth[:3,3])*1000)
    fit['diagnostic_after_error_mm']=float(np.linalg.norm(np.array(fit['position'])-truth[:3,3])*1000)
    fit['diagnostic_angle_deg']=float(Rotation.from_matrix(truth[:3,:3].T@np.array(fit['rotation'])).magnitude()*180/np.pi)
    segments=np.load(ROOT/'experiments/configs/sc_front_rim.npz')['segments_m']
    for view in views:
        img=view['image'].copy()
        world=(np.array(fit['rotation'])@segments.reshape(-1,3).T).T+fit['position']
        uv=project(view['P'],world).reshape(-1,2,2)
        for segment in uv:cv2.polylines(img,[np.rint(segment).astype(np.int32)],False,(0,255,255),1)
        cv2.imwrite(str(a.out/f'{view["camera"]}.png'),img)
    (a.out/'diagnostic_scores.json').write_text(json.dumps(fit,indent=2)+'\n');print(json.dumps(fit,indent=2))


if __name__=='__main__':main()

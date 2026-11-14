"""Bounded visual SC front-rim refinement from public CAD and calibrated RGB."""
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
TEMPLATE = Path(__file__).with_name('sc_front_rim.npz')


def project(P, points):
    q = (P @ np.c_[points, np.ones(len(points))].T).T
    return q[:, :2]/q[:, 2:]


def edge_distance(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, np.array([90,80,60]), np.array([130,255,255]))
    edge = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50, 100)
    edge &= cv2.dilate(blue, np.ones((3,3), np.uint8))
    white = (hsv[:,:,1] < 60) & (hsv[:,:,2] > 80)
    # A held cable can cross the target even after a camera retreat. Its
    # blue/orange boundary is an occlusion edge, not a CAD front-rim edge.
    # Reuse the body fitter's cable colour class; no pose gate is weakened.
    orange = cv2.inRange(hsv, np.array([5,100,60]), np.array([35,255,255])) != 0
    occluded = (white | orange).astype(np.uint8)*255
    occluded = cv2.dilate(occluded, np.ones((5,5), np.uint8))
    edge[occluded != 0] = 0
    return cv2.distanceTransform(255-edge, cv2.DIST_L2, cv2.DIST_MASK_PRECISE).astype(float), occluded


def fit_rim(views, position, rotation, *, raster_offset_px=0., full_rotation=False, prepared_edges=None):
    points = np.load(TEMPLATE)['points_m']
    prepared = []; occlusion_maps = []
    for view in views:
        distance, occluded = (edge_distance(view['image']) if prepared_edges is None
                              else prepared_edges[id(view)])
        initial_uv = project(view['P'], (rotation @ points.T).T+position)+raster_offset_px
        visible = map_coordinates(occluded, initial_uv[:, ::-1].T, order=0, mode='constant', cval=255) == 0
        prepared.append((view, distance, visible))
        occlusion_maps.append(occluded)
    if any(np.sum(visible)<30 for _,_,visible in prepared):
        return {'optimizer_success':False, 'mean_edge_distance_px':None,
                'visible_points_per_view':[int(np.sum(v)) for _,_,v in prepared]}
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
    # The initial learned pose can put a visible physical edge inside the
    # plug's image mask. Recompute visibility after correcting the pose,
    # otherwise the primary and leave-one-out fits compare different edges.
    visibility_iterations=0
    for _ in range(3):
        pos,R=pose(best.x); world=(R@points.T).T+pos
        updated=[]
        for (view,field,visible),occluded in zip(prepared,occlusion_maps):
            uv=project(view['P'],world)+raster_offset_px
            visible=map_coordinates(occluded,uv[:,::-1].T,order=0,mode='constant',cval=255)==0
            updated.append((view,field,visible))
        if all(np.array_equal(a[2],b[2]) for a,b in zip(prepared,updated)):
            break
        prepared=updated; visibility_iterations+=1
        if any(np.sum(v)<30 for _,_,v in prepared):
            return {'optimizer_success':False,'mean_edge_distance_px':None,
                    'visible_points_per_view':[int(np.sum(v)) for _,_,v in prepared]}
        best=least_squares(residual,best.x,bounds=([-4,-4,-4]+[-10]*angles,[4,4,4]+[10]*angles),
                           loss='soft_l1',f_scale=1,max_nfev=100,diff_step=1e-4)
    pos,R=pose(best.x); r=residual(best.x)
    return {'position': pos.tolist(), 'rotation': R.tolist(), 'delta_mm_deg': best.x.tolist(),
            'mean_edge_distance_px': float(np.mean(r)), 'median_edge_distance_px': float(np.median(r)),
            'p90_edge_distance_px': float(np.percentile(r,90)), 'cost': float(best.cost),
            'visible_points_per_view': [int(np.sum(v)) for _,_,v in prepared],
            'optimizer_success': bool(best.success), 'raster_offset_px':raster_offset_px,
            'full_rotation':full_rotation,'visibility_iterations':visibility_iterations}


def refine_with_agreement(views, position, rotation, *, raster_offset_px=0.):
    """Three views and agreement of all leave-one-view-out fits are required.

    The board-normal prior is inherited from the original detector. View
    agreement is an abstention test, not calibrated pose uncertainty.
    """
    if len(views) != 3:
        return None, {'reason':'three_views_required'}
    # Image-local reuse; no pose, exposure transform or fitted result is cached.
    prepared_edges={id(view):edge_distance(view['image']) for view in views}
    result=fit_rim(views,position,rotation,raster_offset_px=raster_offset_px,prepared_edges=prepared_edges)
    if (not result['optimizer_success'] or min(result['visible_points_per_view']) < 30
            or result['mean_edge_distance_px'] > 2.
            or np.linalg.norm(result['delta_mm_deg'][:3]) > 4.
            or abs(result['delta_mm_deg'][3]) >= 9.9):
        return None, dict(result,reason='fit_quality_rejected')
    points=np.load(TEMPLATE)['points_m']
    primary=(np.array(result['rotation'])@points.T).T+result['position']
    disagreements=[]
    for omitted in range(3):
        subset=[v for i,v in enumerate(views) if i!=omitted]
        alternate=fit_rim(subset,np.array(result['position']),np.array(result['rotation']),raster_offset_px=raster_offset_px,prepared_edges=prepared_edges)
        if not alternate['optimizer_success'] or alternate['mean_edge_distance_px']>2.:
            return None, dict(result,reason='held_view_fit_failed',disagreements_m=disagreements)
        model=(np.array(alternate['rotation'])@points.T).T+alternate['position']
        disagreements.append(float(np.max(np.linalg.norm(model-primary,axis=1))))
    diagnostics=dict(result,reason='accepted',disagreements_m=disagreements)
    if max(disagreements)>.00025:
        return None,dict(diagnostics,reason='held_view_disagreement')
    return (np.array(result['position']),np.array(result['rotation'])),diagnostics


def modeled_displacement(before,after):
    points=np.load(TEMPLATE)['points_m']
    a=(before[1]@points.T).T+before[0]
    b=(after[1]@points.T).T+after[0]
    return float(np.max(np.linalg.norm(a-b,axis=1)))




"""Prepare bounded-distance ROI acceleration, without editing the active fit."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'.artifacts/repair36/rim_roi_candidate'
HELPER='''def distance_field_roi(P,nominal_world,local_radius,image_shape):
    """Conservative projection bound for the entire existing fit domain.

    Translation has norm <=sqrt(3)*4 mm and yaw perturbation <=5 degrees.
    For a perspective quotient, the exact displacement bound is
    r*||P_j-u_j P_2||/(q_2-r*||P_2||). Add 14 px for the original 12 px
    clipped distance and interpolation support, so excluded image edges
    cannot change a visible residual below its existing clip.
    """
    q=(P@np.c_[nominal_world,np.ones(len(nominal_world))].T).T
    radius=np.sqrt(3)*.004+2*local_radius*np.sin(np.deg2rad(5)/2)
    denominator=q[:,2]-radius*np.linalg.norm(P[2,:3])
    height,width=image_shape
    if np.any(denominator<=0):return (0,0,width,height)
    uv=q[:,:2]/q[:,2:]
    bound=np.column_stack([radius*np.linalg.norm(P[j,:3]-uv[:,j,None]*P[2,:3],axis=1)/denominator for j in range(2)])
    lo=np.floor(np.min(uv-bound,axis=0)-14).astype(int)
    hi=np.ceil(np.max(uv+bound,axis=0)+14).astype(int)+1
    x0,y0=max(0,int(lo[0])),max(0,int(lo[1]))
    x1,y1=min(width,int(hi[0])),min(height,int(hi[1]))
    if x1-x0<2 or y1-y0<2:return (0,0,width,height)
    return x0,y0,x1,y1


'''


def main():
    OUT.mkdir(exist_ok=False)
    source=ROOT/'aic_model/aic_model/insertion/sfp_fixture_refinement.py';text=source.read_text()
    changes=[('def refine_rim(views,seed,plug):',HELPER+'def refine_rim(views,seed,plug):'),
        ("    if seed['slot']==1:edges=[e-SLOT1 for e in edges]", "    if seed['slot']==1:edges=[e-SLOT1 for e in edges]\n    local_radius=max(np.linalg.norm(e,axis=1).max() for e in edges)"),
        ('        for points in edges:', '        x0,y0,x1,y1=distance_field_roi(P,(R0@np.vstack(edges).T).T+p0,local_radius,gray.shape)\n        for points in edges:'),
        ('cv2.distanceTransform(255-edge,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)',
         'cv2.distanceTransform((255-edge)[y0:y1,x0:x1],cv2.DIST_L2,cv2.DIST_MASK_PRECISE)'),
        ('        prepared.append((P,mask,fields))','        prepared.append((P,mask,fields,np.array([x0,y0])))'),
        ('        for P,mask,fields in prepared:','        for P,mask,fields,field_origin in prepared:'),
        ('values=map_coordinates(field,uv[:,::-1].T,','values=map_coordinates(field,(uv-field_origin)[:,::-1].T,')]
    candidate=text
    for before,after in changes:
        if candidate.count(before)!=1:raise RuntimeError('ROI source anchor mismatch: '+before)
        candidate=candidate.replace(before,after)
    (OUT/'sfp_fixture_full.py').write_bytes(source.read_bytes())
    path=OUT/'sfp_fixture_roi.py';path.write_text(candidate,encoding='utf-8')
    (OUT/'candidate.json').write_text(json.dumps(dict(production_changed=False,
        original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        candidate_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        parameter_bounds_changed=False,starts=27,clip_px=12,acceptance_px=2),indent=2)+'\n',encoding='utf-8')
    print(path)


if __name__=='__main__':main()

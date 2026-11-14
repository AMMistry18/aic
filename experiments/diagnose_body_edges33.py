"""Development-only CAD silhouette edge translation; save fits before GT scoring."""
import gzip
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'aic_model'))
from aic_model.insertion import sc_body_refinement as body


def edge_data(view,points,position,rotation):
    image=view['image'];h,w=image.shape[:2]
    uv=body.project(view['P'],(rotation@points.T).T+position)-.5
    indices=cv2.convexHull(uv.astype(np.float32),returnPoints=False).ravel()
    hull=uv[indices];model=points[indices]
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    orange=cv2.inRange(hsv,np.array([5,100,60]),np.array([35,255,255]))
    mask=cv2.dilate(orange|view['tool_mask'],np.ones((7,7),np.uint8))!=0
    rows=[]
    for j in range(len(hull)):
        a=hull[j];b=hull[(j+1)%len(hull)];delta=b-a;length=np.linalg.norm(delta)
        if length<6:continue
        normal=np.array([delta[1],-delta[0]])/length
        mid=(a+b)*.5
        if normal@(mid-hull.mean(0))<0:normal=-normal
        for f in np.linspace(.15,.85,max(2,int(length/5))):
            q=a+f*delta;world=model[j]+f*(model[(j+1)%len(hull)]-model[j])
            xy=np.rint(q).astype(int)
            if not (10<=xy[0]<w-10 and 10<=xy[1]<h-10) or mask[xy[1],xy[0]]:continue
            s=np.linspace(-7,7,141);query=q+s[:,None]*normal
            colors=np.stack([map_coordinates(image[:,:,c].astype(float),query[:,::-1].T,order=1,mode='nearest') for c in range(3)],axis=1)
            fg=np.median(colors[:12],axis=0);bg=np.median(colors[-12:],axis=0);contrast=fg-bg
            norm=contrast@contrast
            if norm<30**2 or np.mean(fg)<60:continue
            alpha=(colors-bg)@contrast/norm
            transitions=np.flatnonzero((alpha[:-1]>=.5)&(alpha[1:]<.5))
            if not len(transitions):continue
            k=min(transitions,key=lambda k:abs(s[k]));offset=s[k]+(.5-alpha[k])*(s[k+1]-s[k])/(alpha[k+1]-alpha[k])
            edge=q+normal*offset;xy=np.rint(edge).astype(int)
            if mask[xy[1],xy[0]]:continue
            rows.append((world,normal,float(normal@edge)))
    return rows


def fit(views,position,rotation):
    points,_=body.template();prior=np.array(position);position=prior.copy()
    for _ in range(3):
        prepared=[edge_data(v,points,position,rotation) for v in views]
        if any(len(x)<8 for x in prepared):return dict(accepted=False,reason='insufficient_edges',counts=list(map(len,prepared)))
        def optimize(selected):
            def residual(x):
                result=[]
                for i in selected:
                    data=prepared[i];pts=np.array([r[0] for r in data]);normal=np.array([r[1] for r in data]);target=np.array([r[2] for r in data])
                    pix=body.project(views[i]['P'],(rotation@pts.T).T+prior+x*.001)-.5
                    result.extend(np.sum(pix*normal,axis=1)-target)
                return np.array(result)
            opt=least_squares(residual,(position-prior)*1000,bounds=([-3]*3,[3]*3),loss='soft_l1',f_scale=.5,max_nfev=100,diff_step=1e-4)
            return opt,float(np.mean(abs(residual(opt.x))))
        opt,error=optimize(range(3));new=prior+opt.x*.001
        if np.linalg.norm(new-position)<1e-7:position=new;break
        position=new
    loo=[float(np.linalg.norm(optimize([i for i in range(3) if i!=omit])[0].x-opt.x)*.001) for omit in range(3)]
    return dict(position=position.tolist(),accepted=error<=1 and max(loo)<=.00025,
                mean_support_error_px=error,leave_one_view_out_disagreement_m=loo,counts=list(map(len,prepared)))


def main():
    summary=json.loads((ROOT/'experiments/evidence/original_repair/persistent_screen32v1.json').read_text());rows=[]
    for run in summary['runs']:
        folder=ROOT/'.artifacts/lastinch_runs'/run['run']/run['run']/'body_inputs/000';d=json.loads((folder/'inputs.json').read_text())
        views=[dict(camera=v['camera'],image=cv2.imread(str(folder/(v['camera']+'.png'))),P=np.array(v['P']),
                    tool_mask=cv2.imread(str(folder/(v['camera']+'_tool_mask.png')),0)) for v in d['views']]
        result=fit(views,np.array(d['initial_position']),np.array(d['initial_rotation']))
        rows.append(dict(run=run['run'],stamp=max(v['stamp_s'] for v in d['views']),**result))
    out=ROOT/'.artifacts/body_edges33';out.mkdir(exist_ok=True)
    (out/'fits_without_truth.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
    base=np.eye(4);base[:3,:3]=np.diag([-1.,-1.,1.]);base[:3,3]=[-.2,.2,0]
    for r in rows:
        if 'position' not in r:continue
        folder=ROOT/'.artifacts/lastinch_runs'/r['run']/r['run']
        trace=json.load(gzip.open(folder/'truth_trace.json.gz','rt'));actual=min(trace,key=lambda t:abs(t['sim_time_s']-r['stamp']))
        truth=np.linalg.inv(base)@np.array(actual['tip_world']);r['diagnostic_error_base_mm']=(1000*(np.array(r['position'])-truth[:3,3])).tolist()
    (out/'diagnostic_scores.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
    for r in rows:print(json.dumps(r))


if __name__=='__main__':main()

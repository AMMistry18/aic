"""Actual paired image crops and observed foreground masks; no inferred pose."""
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'aic_model')]
from experiments.lastinch_geometry import HOME,robot_fk
from experiments.sc_foreground_refinement30 import suppress_context


def main():
    name='dev_3000_cable30v1_original_perception_rim_position_body_track_handoff';r=ROOT/'.artifacts/lastinch_runs'/name/name
    rows=json.loads((ROOT/'.artifacts/reference30/foreground_anchor30_last/fits_without_truth.json').read_text())['rows'];records=[x for x in rows if x.get('run')==name]
    fk=robot_fk(ET.parse(ROOT/'.artifacts/lastinch_resources/ur5e.urdf').getroot(),HOME)
    canvas=Image.new('RGB',(1280,1020),'#151515');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18);hashes={}
    for i,camera in enumerate(['left_camera','center_camera','right_camera']):
        d0=next(x for x in records[0]['preprocessing'] if x['camera']==camera);box=np.array(d0['refinement_proposal_box']);x0,y0,x1,y1=d0['proposal_roi_xyxy'];C=np.linalg.inv(fk[camera+'/optical'])@fk['gripper/tcp']
        pairs=[]
        for capture in ['perception_before','perception_after']:
            path=r/capture/(camera+'_raw.png');image=cv2.imread(str(path));hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest();metadata=json.loads((r/capture/'camera_geometry.json').read_text());K=np.array(next(v for v in metadata if v['camera']==camera)['K'])
            masked,diagnostic=suppress_context(image,box,K,C);pairs.append((image,masked,diagnostic))
        for j,(image,label) in enumerate([(pairs[0][0],'Raw before'),(pairs[1][0],'Raw after'),(pairs[0][1],'Masked before'),(pairs[1][1],'Masked after')]):
            crop=Image.fromarray(cv2.cvtColor(image[y0:y1,x0:x1],cv2.COLOR_BGR2RGB));crop.thumbnail((305,285));canvas.paste(crop,(j*320+(320-crop.width)//2,i*330+48));draw.text((j*320+10,i*330+5),camera+' | '+label,font=font,fill='white')
        draw.text((10,i*330+312),f"Observed support pixels: before {pairs[0][2]['observed_foreground_pixels']}; after {pairs[1][2]['observed_foreground_pixels']}. Identical RGB-proposal ROI in both.",font=font,fill='#ffcc66')
    out=ROOT/'experiments/evidence/original_repair';canvas.save(out/'rigid_grasp_actual_image_pair.png');(out/'rigid_grasp_actual_image_pair_sources.json').write_text(json.dumps(dict(run=name,source_sha256=hashes,preprocessed_panels_explicitly_labeled=True,not_insertion_success_evidence=True),indent=2)+'\n')


if __name__=='__main__':main()

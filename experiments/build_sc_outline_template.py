"""Extract public SC mesh outline vertices in its physical mouth frame.

Only source asset geometry/materials are read. No scene placement or images.
Trimesh is an offline extraction dependency; runtime uses the resulting NPZ.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
import trimesh
from experiments.lastinch_geometry import ROOT,MODELS,sdf_frames,source_model


def main():
    path=MODELS/'SC Port 2/sc_port_visual.glb'
    scene=trimesh.load_scene(path)
    vertices=[]
    for node in scene.graph.nodes_geometry:
        transform,name=scene.graph[node];mesh=scene.geometry[name]
        if mesh.visual.material.name!='Material.004':continue
        rgba=mesh.visual.to_color().vertex_colors.astype(float)
        blue=(rgba[:,2]>rgba[:,0]*1.3)&(rgba[:,2]>rgba[:,1]*1.1)
        vertices.append(trimesh.transform_points(mesh.vertices[blue],transform))
    body=np.concatenate(vertices)
    frame=sdf_frames(source_model('SC Port 2'))['sc_port_base_link_entrance']
    mouth=(np.linalg.inv(frame)@np.column_stack([body,np.ones(len(body))]).T).T[:,:3]
    hull=ConvexHull(mouth)
    points=mouth[hull.vertices]
    out=ROOT/'experiments/configs/sc_blue_outline.npz'
    np.savez(out,points_m=points)
    metadata={'purpose':'public mesh geometry in mouth frame; no scene placement',
              'source_asset':'SC Port 2/sc_port_visual.glb','source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'source_model_sha256':hashlib.sha256((MODELS/'SC Port 2/model.sdf').read_bytes()).hexdigest(),
              'material':'Material.004','blue_rule':'B > 1.3*R and B > 1.1*G on source texture vertices',
              'mouth_from_body':np.linalg.inv(frame).tolist(),'source_blue_vertex_count':len(body),
              'hull_vertex_count':len(points),'trimesh_version':trimesh.__version__,
              'npz_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
    out.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'vertices':len(points),'bounds_m':[points.min(0).tolist(),points.max(0).tolist()]}))


if __name__=='__main__':main()

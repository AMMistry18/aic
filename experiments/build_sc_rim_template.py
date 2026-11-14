"""Extract front-face boundary edges from the public SC visual mesh."""
import hashlib
import json
from pathlib import Path
import numpy as np
import trimesh
from experiments.lastinch_geometry import ROOT, MODELS, source_model, sdf_frames


def main():
    path = MODELS/'SC Port 2/sc_port_visual.glb'
    scene = trimesh.load_scene(path)
    inv = np.linalg.inv(sdf_frames(source_model('SC Port 2'))['sc_port_base_link_entrance'])
    triangles = []
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]; mesh = scene.geometry[name]
        if mesh.visual.material.name != 'Material.004': continue
        rgba = mesh.visual.to_color().vertex_colors.astype(float)
        blue = (rgba[:, 2] > rgba[:, 0]*1.3) & (rgba[:, 2] > rgba[:, 1]*1.1)
        vertices = trimesh.transform_points(mesh.vertices, inv @ transform)
        faces = mesh.faces[np.all(blue[mesh.faces], axis=1)]
        t = vertices[faces]
        # The foremost planar rim is a visible, physical surface, not the rear
        # wall of the hole. All choices depend only on the public source mesh.
        front = np.min(vertices[blue, 2])
        keep = np.max(t[:, :, 2], axis=1) < front + .00001
        triangles.extend(t[keep])
    triangles = np.asarray(triangles)
    vertices, inverse = np.unique(np.round(triangles.reshape(-1, 3), 7), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    unique, count = np.unique(edges, axis=0, return_counts=True)
    segments = vertices[unique[count == 1]]
    segments = segments[np.linalg.norm(segments[:, 1]-segments[:, 0], axis=1) > .0001]
    points = np.concatenate([np.linspace(a, b, max(3, int(np.linalg.norm(a-b)/.00025)+1)) for a, b in segments])
    out = ROOT/'experiments/configs/sc_front_rim.npz'
    np.savez(out, points_m=points, segments_m=segments)
    info = {'purpose': __doc__, 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'triangles': len(triangles), 'segments': len(segments), 'points': len(points),
            'bounds_m': [points.min(0).tolist(), points.max(0).tolist()], 'scene_state_used': False}
    out.with_suffix('.json').write_text(json.dumps(info, indent=2)+'\n')
    print(json.dumps(info, indent=2))


if __name__ == '__main__': main()

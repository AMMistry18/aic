"""Paper setup figure from actual headless camera captures, no generated imagery."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]


def main():
    name='dev_41000_translation30v1_original_perception_rim_position_body_track_noangular_handoff';r=ROOT/'.artifacts/lastinch_runs'/name/name
    out=ROOT/'experiments/evidence/original_repair';sources={}
    with plt.rc_context({'font.size':8,'svg.fonttype':'none'}):
        fig,axes=plt.subplots(1,3,figsize=(7.1,2.35),layout='constrained')
        for ax,camera,label in zip(axes,['overview','board','target'],['(a) UR5e and task scene','(b) Fixture and cable context','(c) Verified held SC seat']):
            files=sorted(r.glob(camera+'_*.jpg'),key=lambda p:float(p.stem.rsplit('_',1)[1]));source=files[-1]
            ax.imshow(plt.imread(source));ax.set_axis_off();ax.set_title(label,fontsize=8.5)
            sources[camera]=dict(path=str(source.relative_to(ROOT)),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),capture_sim_s=float(source.stem.rsplit('_',1)[1]))
        fig.savefig(out/'actual_headless_setup30.png',dpi=240);fig.savefig(out/'actual_headless_setup30.svg');plt.close(fig)
    (out/'actual_headless_setup30_sources.json').write_text(json.dumps(dict(run=name,actual_camera_images_only=True,upstream_angular_alignment_assumed=True,grasp_is_prescribed_rigid=True,post_release_retention_not_tested=True,sources=sources),indent=2)+'\n')


if __name__=='__main__':main()

"""Prepare a reviewable candidate without touching the running policy."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('aic_model/aic_model/insertion/sfp_contact_datum.py')
OUT=ROOT/'.artifacts/repair35/axial_candidate'


def main():
    OUT.mkdir(exist_ok=False)
    original=(ROOT/SOURCE).read_text(encoding='utf-8')
    additions=[('class DatumController(SfpInsertionController):', '''def waypoint_axial_integral(integral_N,error_m,dt_s,force_N):
    """Bounded own-robot load compensation after the free path reaches its goal."""
    if not np.isfinite([integral_N,error_m,dt_s,force_N]).all() or dt_s<0 or force_N<0:
        raise RuntimeError('Invalid free-waypoint axial observation')
    if force_N>=1.:return integral_N
    return float(np.clip(integral_N+200.*error_m*min(.04,dt_s),-.15,.15))


class DatumController(SfpInsertionController):'''),
    ('''        start=self.clock();last=start;point,_,_,_=self.read();command=point.copy();arrived=None
        while self.clock()-start<budget:''', '''        start=self.clock();last=start;point,_,_,_=self.read();command=point.copy();arrived=None
        integral=0.;axis=self.Rp[:,2]
        record=dict(start_sim_s=start,goal_base_m=goal.tolist(),samples=[],
                    axial_integral_gain_N_per_m_s=200.,axial_integral_bound_N=.15,
                    compensation_scope='free waypoint only; own estimated tip error')
        self.diagnostic.setdefault('waypoints',[]).append(record)
        while self.clock()-start<budget:'''),
    ('''            # Persistent path, bounded elastic offset from the measured robot.
            delta=command-point;target=point+delta*min(1.,.0005/max(np.linalg.norm(delta),1e-12))
            self.send(target,self.held_rotation)''', '''            # Correct residual axial load bias only after the nominal free
            # path reaches its destination. Do not integrate travel damping
            # or use this compensation during either contact measurement.
            if np.linalg.norm(goal-command)<1e-9:
                integral=waypoint_axial_integral(integral,float(axis@(goal-point)),dt,float(np.linalg.norm(force)))
            # Preserve the total 0.5 mm pose-lead bound, including compensation.
            desired=command+axis*(integral/500.)
            delta=desired-point;target=point+delta*min(1.,.0005/max(np.linalg.norm(delta),1e-12))
            record['samples'].append(dict(sim_s=stamp,error_port_m=(self.Rp.T@(goal-point)).tolist(),
                axial_integral_N=integral,pose_lead_m=float(np.linalg.norm(target-point))))
            self.send(target,self.held_rotation)''')]
    candidate=original
    for before,after in additions:
        if candidate.count(before)!=1:raise RuntimeError('Candidate source anchor mismatch')
        candidate=candidate.replace(before,after)
    (OUT/'sfp_contact_datum_v27.py').write_text(original,encoding='utf-8')
    (OUT/'sfp_contact_datum_v28.py').write_text(candidate,encoding='utf-8')
    metadata=dict(production_policy_changed=False,source=str(SOURCE).replace('\\','/'),
        original_sha256=hashlib.sha256((ROOT/SOURCE).read_bytes()).hexdigest(),
        candidate_sha256=hashlib.sha256((OUT/'sfp_contact_datum_v28.py').read_bytes()).hexdigest(),
        required_before_deployment='completed frozen V27 queue, source hash check, candidate unit tests',
        proposed_physical_regressions=['dev_45002','dev_45003'],held_out_final_trials_run=0)
    (OUT/'candidate.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()

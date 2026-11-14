"""Prepare only; never overwrite sources used by an active physical trial."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'.artifacts/repair36/pulse_lip_candidate';OUT.mkdir(exist_ok=True)
source=ROOT/'aic_model/aic_model/insertion/sfp_contact_datum.py'
original=source.read_text()
helper='''def kinematic_lip_candidate(samples):
    if not samples:return None
    end=samples[-1][0];rows=np.asarray([s for s in samples if end-.83<=s[0]<=end])
    if len(rows)<20 or rows[-1,0]-rows[0,0]<.77 or not np.isfinite(rows).all():return None
    gaps=np.diff(rows[:,0]);advance=rows[-1,2]-rows[0,2]
    travel=np.ptp(rows[:,1]);lead=float(np.median(rows[-8:,2]-rows[-8:,1]))
    if np.any(gaps<=0) or gaps.max()>.05 or advance<.0001 or travel>.000025 or lead<.00015:return None
    return dict(command_advance_m=float(advance),observed_travel_m=float(travel),elastic_lead_m=lead)


'''
method='''    def _lip_force_plateau(self,target,force_N):
        self._axial_tracking_force_N=force_N
        start=self.clock();rows=[];points=[];last_sensor=None
        self.send(target,self.held_rotation)
        while self.clock()-start<.65:
            point,_,force,stamp=self.read();sensor=self._last_wrist_stamp_s
            if stamp-start>=.2 and (last_sensor is None or sensor>last_sensor):
                rows.append([sensor,float(self.Rp[:,2]@(point-self.port_pos)),float(-force[2])])
                points.append((self.Rp.T@point).tolist());last_sensor=sensor
            self.send(target,self.held_rotation);self.policy.sleep_for(.02)
        return rows,self.clock(),points

    def mouth_stop(self,offset,*,start_depth_m=-.0015):
        from .qualified_lip_contact import fixed_triplet
        if getattr(self,'_axial_tracking_bound_N',.25)<1.:
            return self._legacy_mouth_stop(offset,start_depth_m=start_depth_m)
        axis=self.Rp[:,2];startpoint=self.port_pos+self.Rp@np.array([offset,0.,start_depth_m])
        self.line(startpoint)
        point,_,_,last=self.read();command=float(axis@(point-self.port_pos));samples=[];start=last
        while self.clock()-start<28:
            point,_,f,stamp=self.read();depth=float(axis@(point-self.port_pos))
            if not samples or stamp>samples[-1][0]:samples.append([stamp,depth,command,float(-f[2])])
            self.diagnostic['axial_probe_samples'].append(dict(sim_s=stamp,offset_m=offset,
                measured_depth_m=depth,applied_command_m=command,force_port_N=f.tolist()))
            trigger=kinematic_lip_candidate(samples)
            if trigger:
                # Blocking is a trigger, never an accepted contact measurement.
                # One fixed 0/1/0 N excitation with actual unique wrench stamps.
                target=self.port_pos+self.Rp@np.array([offset,0.,command])
                try:
                    low,t0,p0=self._lip_force_plateau(target,0.)
                    high,t1,p1=self._lip_force_plateau(target,1.)
                    released,t2,p2=self._lip_force_plateau(target,0.)
                finally:self._axial_tracking_force_N=0.
                proof=fixed_triplet(low,high,released,low_end=t0,high_end=t1,release_end=t2,kinematic_blocked=True)
                points=np.asarray(p0+p1+p2)
                stationary=bool(len(points) and np.all(np.ptp(points,axis=0)<=np.array([.00005,.00005,.000025])))
                record=dict(offset_m=offset,trigger=trigger,low=low,high=high,released=released,
                            proof=proof,own_tip_stationary_across_excitation=stationary,
                            object_or_target_truth_used=False)
                self.diagnostic.setdefault('qualified_lip_excitations',[]).append(record)
                if not proof['accepted'] or not stationary:
                    raise RuntimeError('SFP fixed lip excitation unobservable: '+proof['reason'])
                return dict(**trigger,offset_m=offset,stop_depth_m=proof['low']['depth_m'],
                            reaction_increase_N=proof['contrast_N'],qualified_reversible_excitation=True)
            command=min(command+.0004*min(.04,max(0.,stamp-last)),depth+.0003,.003)
            last=stamp
            if command>=.003 and depth>=command-.00003:raise RuntimeError('No mouth lip in bounded axial capture range')
            self.send(self.port_pos+self.Rp@np.array([offset,0.,command]),self.held_rotation)
            self.policy.sleep_for(.02)
        raise RuntimeError('SFP mouth stop unobservable')

'''
candidate=original.replace('class DatumController(SfpInsertionController):',helper+'class DatumController(SfpInsertionController):')
start=candidate.index('    def mouth_stop(');end=candidate.index('    def run_datum(',start)
legacy=candidate[start:end].replace('    def mouth_stop(','    def _legacy_mouth_stop(',1)
candidate=candidate[:start]+legacy+method+candidate[end:]
(OUT/'sfp_contact_datum.py').write_text(candidate)
(OUT/'qualified_lip_contact.py').write_text((ROOT/'experiments/pulse_contact36_candidate.py').read_text())
(OUT/'candidate.json').write_text(json.dumps({'installed':False,'based_on_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'candidate_sha256':hashlib.sha256((OUT/'sfp_contact_datum.py').read_bytes()).hexdigest(),
    'scope':'new SFP 20 N profile only; no simulator truth; fixed reversible force excitation'},indent=2)+'\n')
print(OUT)

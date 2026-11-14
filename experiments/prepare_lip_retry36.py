"""Prepare a contact retry candidate while the clearance-only trial is frozen."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'.artifacts/repair36/lip_retry_candidate'

METHOD='''    def mouth_stop(self,offset,*,start_depth_m=-.0015):
        axis=self.Rp[:,2];startpoint=self.port_pos+self.Rp@np.array([offset,0.,start_depth_m])
        self.line(startpoint)
        point,_,_,last=self.read();command=float(axis@(point-self.port_pos));samples=[];start=last
        phase='approach';release_reference=None;retry_record=None
        while self.clock()-start<28:
            point,_,f,stamp=self.read();depth=float(axis@(point-self.port_pos))
            dt=min(.04,max(0.,stamp-last));last=stamp
            self.diagnostic['axial_probe_samples'].append(dict(sim_s=stamp,offset_m=offset,phase=phase,
                measured_depth_m=depth,applied_command_m=command,force_port_N=f.tolist()))
            if phase=='unload':
                command=max(release_reference-.0002,command-.0002*dt)
                if depth<=release_reference-.00012:
                    retry_record['released_distance_m']=release_reference-depth
                    retry_record['release_sim_s']=stamp
                    phase='reload';samples=[]
            else:
                if not samples or stamp>samples[-1][0]:samples.append([stamp,depth,command,float(-f[2])])
                stop=blocked_window(samples)
                if stop:
                    stop.update(offset_m=offset,unload_reload_retry_used=retry_record is not None)
                    return stop
                if phase=='approach' and kinematic_block_candidate(samples):
                    # Motion clipping is only a trigger to gather a better
                    # measurement, never permission to accept a contact datum.
                    # One measured release and a gentler reload remove the
                    # initial impact transient from the new observation window.
                    release_reference=depth;phase='unload';samples=[]
                    retry_record=dict(offset_m=offset,start_sim_s=stamp,blocked_depth_m=depth,
                        release_goal_m=.0002,minimum_measured_release_m=.00012,
                        reload_speed_m_s=.0002,contact_accepted_from_kinematics=False)
                    self.diagnostic.setdefault('lip_retries',[]).append(retry_record)
                else:
                    speed=.0004 if phase=='approach' else .0002
                    command=min(command+speed*dt,depth+.0003,.003)
            if command>=.003 and depth>=command-.00003:
                raise RuntimeError('No mouth lip in bounded axial capture range')
            self.send(self.port_pos+self.Rp@np.array([offset,0.,command]),self.held_rotation)
            self.policy.sleep_for(.02)
        raise RuntimeError('SFP mouth stop unobservable')

'''

HELPER='''def kinematic_block_candidate(samples):
    """Provisional trigger only; the original reaction gate stays mandatory."""
    if not samples:return False
    end=samples[-1][0];rows=np.asarray([s for s in samples if end-.83<=s[0]<=end])
    if len(rows)<20 or rows[-1,0]-rows[0,0]<.77 or not np.isfinite(rows).all():return False
    gaps=np.diff(rows[:,0])
    if np.any(gaps<=0) or np.max(gaps)>.05:return False
    return bool(rows[-1,2]-rows[0,2]>=.0001 and np.ptp(rows[:,1])<=.000025
                and np.median(rows[-8:,2]-rows[-8:,1])>=.00015)


'''


def main():
    OUT.mkdir(exist_ok=False)
    source=ROOT/'aic_model/aic_model/insertion/sfp_contact_datum.py'
    text=source.read_text();begin=text.index('    def mouth_stop(');end=text.index('    def run_datum(',begin)
    candidate=(text[:begin]+METHOD+text[end:]).replace('class DatumController(SfpInsertionController):',HELPER+'class DatumController(SfpInsertionController):')
    (OUT/'sfp_contact_datum_v29.py').write_bytes(source.read_bytes())
    path=OUT/'sfp_contact_datum_v30.py';path.write_text(candidate,encoding='utf-8')
    metadata=dict(production_policy_changed_at_preparation=False,
        original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        candidate_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        contact_thresholds_changed=False,one_retry_only=True,simulator_truth_inputs=False)
    (OUT/'candidate.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(metadata))


if __name__=='__main__':main()

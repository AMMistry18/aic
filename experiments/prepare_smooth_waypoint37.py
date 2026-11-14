"""Prepare controller copies only while the original-source regression runs."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'.artifacts/repair37/waypoint_candidate';OUT.mkdir(parents=True,exist_ok=True)
source=ROOT/'aic_model/aic_model/insertion/sfp_contact_datum.py';original=source.read_text()
candidate=original.replace('import os\n','import os\nimport time\n',1)
candidate=candidate.replace("if profile=='fast36':return .005,.001,.5,True", "if profile in ('fast36','fast37'):return .005,.001,.5,True")
candidate=candidate.replace("if profile=='fast36':\n", "if profile in ('fast36','fast37'):\n",1)
candidate=candidate.replace('        parameters=motion_parameters(profile)\n','        parameters=motion_parameters(profile)\n        self._motion_profile=profile\n',1)
stop='''    def _stop_motion_compensation(self):
        # A deadline hold must not inherit travel/probe feedforward or PI.
        # Calibrated payload support belongs to the outer harness and remains.
        self._tracking_motion_stopped=True
        self._axial_tracking_force_N=0.
        self._path_damping_force_base_N=np.zeros(3)
        self.diagnostic['motion_compensation_stopped']=True

'''
candidate=candidate.replace('    def guard(self):\n',stop+'''    def guard(self):
        deadline=getattr(self.policy,'_action_deadline_wall',None)
        if deadline is not None and time.monotonic()>deadline:
            self._stop_motion_compensation()
            self.diagnostic['reason']='action_deadline'
''',1)
candidate=candidate.replace('        except RuntimeError as exc:\n            self._axial_tracking_force_N=0.\n',
    '        except RuntimeError as exc:\n            self._stop_motion_compensation()\n',1)
old='        try:return self._line_tracking(goal,speed=speed,settle=settle,budget=budget)\n'
new='''        try:
            if getattr(self,'_motion_profile','legacy36')=='fast37' and speed==.002:
                return self._line_smooth(goal,settle=settle,budget=budget)
            return self._line_tracking(goal,speed=speed,settle=settle,budget=budget)
'''
assert old in candidate;candidate=candidate.replace(old,new,1)
method='''    def _line_smooth(self,goal,*,settle=.3,budget=12):
        from .waypoint_motion import smooth_waypoint
        start=self.clock();last=start;initial,_,_,_=self.read();arrived=None;integral=0.
        # AIC Cartesian errors and impedance matrices are in base_link.
        # Only the resulting feedforward vector is rotated into TCP at send.
        axis=self.Rp[:,2];damping=np.diag(np.asarray(self.DAMPING,dtype=float)[:3])
        _,_,duration=smooth_waypoint(initial,goal,0.)
        record=dict(start_sim_s=start,goal_base_m=goal.tolist(),samples=[],
                    axial_integral_gain_N_per_m_s=500.,axial_integral_bound_N=.15,
                    trajectory='bounded quintic 5mm/s, 20mm/s^2; damping feedforward',
                    scheduled_duration_s=duration,compensation_scope='own robot tracking only')
        self.diagnostic.setdefault('waypoints',[]).append(record)
        try:
            while self.clock()-start<budget:
                point,_,force,stamp=self.read();dt=min(.04,max(0.,stamp-last));last=stamp
                command,velocity,_=smooth_waypoint(initial,goal,stamp-start)
                integral=waypoint_axial_integral(integral,float(axis@(command-point)),dt,float(np.linalg.norm(force)))
                self._axial_tracking_force_N=integral
                self._path_damping_force_base_N=damping@velocity
                delta=command-point;target=point+delta*min(1.,.0005/max(np.linalg.norm(delta),1e-12))
                record['samples'].append(dict(sim_s=stamp,error_port_m=(self.Rp.T@(goal-point)).tolist(),
                    axial_integral_N=integral,pose_lead_m=float(np.linalg.norm(target-point)),
                    scheduled_velocity_base_m_s=velocity.tolist(),damping_force_base_N=self._path_damping_force_base_N.tolist()))
                self.send(target,self.held_rotation)
                if (stamp-start>=duration or np.linalg.norm(goal-initial)<.00004) and np.linalg.norm(goal-point)<.00004:
                    if arrived is None:arrived=stamp
                    if stamp-arrived>=settle:return
                else:arrived=None
                self.policy.sleep_for(.02)
            raise RuntimeError('SFP datum waypoint tracking timeout')
        finally:
            self._path_damping_force_base_N=np.zeros(3);self._axial_tracking_force_N=0.

'''
candidate=candidate.replace('    def _line_tracking(self,goal,*,speed=.002,settle=.3,budget=12):',method+'    def _line_tracking(self,goal,*,speed=.002,settle=.3,budget=12):',1)
(OUT/'sfp_contact_datum.py').write_text(candidate)
(OUT/'waypoint_motion.py').write_text((ROOT/'experiments/waypoint_motion37_candidate.py').read_text())
tracking=ROOT/'aic_model/aic_model/insertion/tracking_compensation.py';text=tracking.read_text()
anchor='            extra=bounded_axial_feedforward(extra,controller.Rp[:,2],\n'
replace='''            path=np.asarray(getattr(controller,'_path_damping_force_base_N',np.zeros(3)))
            if path.shape!=(3,) or not np.isfinite(path).all() or np.linalg.norm(path)>.5+1e-12:
                raise RuntimeError('Invalid free-path damping feedforward')
            extra=bounded_axial_feedforward(extra+path,controller.Rp[:,2],
'''
assert anchor in text
text=text.replace(anchor,replace,1)
start=text.index('            extra=tracker.update(');end=text.index('            tcp_force=',start)
moving=text[start:end]
text=text[:start]+'''            if getattr(controller,'_tracking_motion_stopped',False):
                extra=np.zeros(3)
            else:
'''+''.join('    '+line if line.strip() else line for line in moving.splitlines(keepends=True))+text[end:]
(OUT/'tracking_compensation.py').write_text(text)
(OUT/'candidate.json').write_text(json.dumps(dict(installed=False,base_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    tracking_base_sha256=hashlib.sha256(tracking.read_bytes()).hexdigest(),
    unchanged_limits=['40um waypoint arrival','0.3s dwell','0.5mm pose lead','0.15N integral','6N inner guard','12s waypoint budget'],
    profile='fast37: 20N SFP only, legacy/fast36 paths retained'),indent=2)+'\n')
print(OUT)

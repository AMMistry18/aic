"""Prepare an uninstalled speed candidate while the baseline stays frozen."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/recovery_20261112/speed43_candidate'


def main():
    changes = {
        'aic_model/aic_model/insertion/waypoint_motion.py': [
            ('def smooth_waypoint(start,goal,elapsed_s):',
             'def smooth_waypoint(start,goal,elapsed_s,*,max_speed=.005,max_acceleration=.020):'),
            ('duration=max(.6,1.875*length/.005,np.sqrt((10/np.sqrt(3))*length/.020))',
             "if (max_speed,max_acceleration) not in ((.005,.020),(.010,.040)):\n        raise RuntimeError('Unqualified waypoint schedule')\n    duration=max(.6,1.875*length/max_speed,np.sqrt((10/np.sqrt(3))*length/max_acceleration))"),
        ],
        'aic_model/aic_model/insertion/contact_registration.py': [
            ('plateau_window_s=.2,wrench_stamp=None):',
             'plateau_window_s=.2,wrench_stamp=None,sweep_time_scale=1.,force_guard_N=6.):'),
            ('initial,rotation,_,_=read();origin=initial.copy()',
             "if (sweep_time_scale,force_guard_N) not in ((1.,6.),(.5,12.)):\n        raise RuntimeError('Unqualified registration motion profile')\n    initial,rotation,_,_=read();origin=initial.copy()"),
            ('np.linalg.norm(force)>6:', 'np.linalg.norm(force)>force_guard_N:'),
            ('segment(center,plus,1.5)', 'segment(center,plus,1.5*sweep_time_scale)'),
            ('segment(plus,minus,3.)', 'segment(plus,minus,3.*sweep_time_scale)'),
            ('segment(minus,center,1.5)', 'segment(minus,center,1.5*sweep_time_scale)'),
        ],
        'aic_model/aic_model/insertion/tracking_compensation.py': [
            ('((.25,.5),(1.,1.2),(1.5,1.75))', '((.25,.5),(1.,1.2),(1.5,1.75),(3.,3.5))'),
        ],
        'aic_model/aic_model/insertion/sfp_contact_datum.py': [
            ("if profile=='fast38':return .010,.002,.5,True",
             "if profile=='fast38':return .010,.002,.5,True\n    if profile=='fast43':return .020,.004,.5,True"),
            ('if force>=3:', "if force>=(6. if profile=='fast43' else 3.):"),
            ("if not np.isfinite([*point,*f]).all() or np.linalg.norm(f)>=6:\n            raise RuntimeError('SFP datum nonfinite/6N force guard')",
             "guard_N=12. if getattr(self,'_motion_profile','legacy36')=='fast43' else 6.\n        if not np.isfinite([*point,*f]).all() or np.linalg.norm(f)>=guard_N:\n            raise RuntimeError(f'SFP datum nonfinite/{guard_N:g}N force guard')"),
            ("in ('fast37','fast38') and speed==.002:", "in ('fast37','fast38','fast43') and speed==.002:"),
            ('_,_,duration=smooth_waypoint(initial,goal,0.)',
             "schedule=dict(max_speed=.010,max_acceleration=.040) if self._motion_profile=='fast43' else {}\n        _,_,duration=smooth_waypoint(initial,goal,0.,**schedule)"),
            ("trajectory='bounded quintic 5mm/s, 20mm/s^2; damping feedforward',",
             "trajectory=('bounded quintic 10mm/s, 40mm/s^2' if schedule else 'bounded quintic 5mm/s, 20mm/s^2')+'; damping feedforward',"),
            ('smooth_waypoint(initial,goal,stamp-start)', 'smooth_waypoint(initial,goal,stamp-start,**schedule)'),
            ('command=min(command+.0004*min(.04,max(0.,stamp-last)),depth+.0003,.003)',
             "probe_speed=.0008 if self._motion_profile=='fast43' else .0004\n            command=min(command+probe_speed*min(.04,max(0.,stamp-last)),depth+.0003,.003)"),
            ("if profile in ('fast36','fast37','fast38'):", "if profile in ('fast36','fast37','fast38','fast43'):"),
            ("self._axial_tracking_bound_N=1.5;self._combined_tracking_bound_N=1.75",
             "self._axial_tracking_bound_N=1.5;self._combined_tracking_bound_N=1.75\n        if profile=='fast43':\n            self._axial_tracking_bound_N=3.;self._combined_tracking_bound_N=3.5"),
            ('plateau_window_s=.4,wrench_stamp=lambda:self._last_wrist_stamp_s)',
             "plateau_window_s=.4,wrench_stamp=lambda:self._last_wrist_stamp_s,\n                sweep_time_scale=.5 if profile=='fast43' else 1.,force_guard_N=12. if profile=='fast43' else 6.)"),
        ],
        'experiments/run_seeded_evaluation.py': [
            ("['legacy36','fast36','fast37','fast38']", "['legacy36','fast36','fast37','fast38','fast43']"),
            ("a.sfp_motion_profile in ('fast36','fast37','fast38')", "a.sfp_motion_profile in ('fast36','fast37','fast38','fast43')"),
        ],
        'experiments/run_original_insertion.py': [
            ("result['harness_revision']=38", "result['harness_revision']=43"),
        ],
        'experiments/run_original_insertion.sh': [
            ('export RL_INSERT_SC_TARGET_FORCE_N=1.5 RL_INSERT_SC_SEAT_FORCE_CAP_N=3.0',
             'export RL_INSERT_SC_TARGET_FORCE_N=3.0 RL_INSERT_SC_SEAT_FORCE_CAP_N=6.0'),
            ('export RL_INSERT_SC_FORCE_ABORT_N=6.0 RL_INSERT_SC_MAX_AXIAL_LEAD_M=0.001',
             'export RL_INSERT_SC_FORCE_ABORT_N=12.0 RL_INSERT_SC_MAX_AXIAL_LEAD_M=0.0015\nexport RL_INSERT_SC_FREE_SPEED_M_S=0.016 RL_INSERT_SC_CONTACT_SPEED_M_S=0.008 RL_INSERT_SC_TERMINAL_SPEED_M_S=0.001'),
        ],
    }
    OUT.mkdir(parents=True,exist_ok=False)
    metadata = {}
    for name, replacements in changes.items():
        original=(ROOT/name).read_bytes(); text=original.decode().replace('\r\n','\n')
        for old,new in replacements:
            if old not in text:raise RuntimeError('Unexpected source: '+name+' '+old)
            text=text.replace(old,new)
        target=OUT/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(text.encode())
        if target.suffix=='.py':compile(text,str(target),'exec')
        metadata[name]=dict(before_sha256=hashlib.sha256(original).hexdigest(),candidate_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (OUT/'changes.json').write_text(json.dumps(dict(installed=False,files=metadata,
        hypothesis='Faster bounded motion between unchanged sensor/observability gates; higher force authority below the same 20N raw ceiling',
        sfp=dict(free_speed_mm_s=20,terminal_speed_mm_s=4,free_waypoint_speed_mm_s=10,
                 probe_speed_mm_s=.8,lateral_sweep_time_scale=.5,travel_pause_N=6,internal_abort_N=12),
        sc=dict(free_speed_mm_s=16,contact_speed_mm_s=8,terminal_speed_mm_s=1,force_cap_N=6,internal_abort_N=12),
        unchanged=['RGB weights','camera synchronization','force excitation 0/1/0 N','contact stationarity/noise thresholds',
                   'force plateau duration','final depth/lateral/angle/contact gates','2000Hz raw force coverage','20N external ceiling'],
        status='unqualified_candidate_requires_development_physics_trials'),indent=2))
    print(json.dumps(dict(prepared_files=len(metadata),installed=False,candidate=str(OUT))))


if __name__=='__main__':main()

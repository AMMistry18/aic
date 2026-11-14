"""Run isolated, matched development trials; never selects held-out cases.

Uses the existing evaluator container, stops only that owned container, and
archives each scene before preparing the next. No global Docker/WSL reset.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'.artifacts';CONTAINER='research-aic-eval'
DEFAULT_METHODS=['initial_pose_guarded','visual_without_recovery','full_visual_force_recovery']
METHODS=DEFAULT_METHODS+['tactile_translation','tactile_full_pose','force_tactile_translation','force_tactile_full_pose','force_tactile_no_registration','original_perception','original_perception_selective','original_perception_reacquire']
METHODS += ['original_perception_sfp_geometry']
METHODS += ['original_perception_sfp_contact_datum_handoff']
METHODS += ['original_perception_rim']
METHODS += ['original_perception_rim_contact']
METHODS += ['original_perception_rim_contact_handoff']
METHODS += ['original_perception_rim_contact_search_handoff']
METHODS += ['original_perception_rim_position_handoff']
METHODS += ['original_perception_rim_position_center_handoff']
METHODS += ['original_perception_rim_position_body_handoff']
METHODS += ['original_perception_rim_position_body_track_handoff']
METHODS += ['original_perception_rim_position_track_handoff']
METHODS += ['original_perception_rim_position_body_track_noangular_handoff']
METHODS += ['original_perception_rim_position_body_noangular_handoff',
            'original_perception_rim_position_track_noangular_handoff']
METHODS += ['original_perception_rim_position_body_track_brake_noangular_handoff']
METHODS += ['original_perception_rim_position_body_track_brake_axial_noangular_handoff']
METHODS += ['original_perception_rim_position_body_track_brake_axial_partial_noangular_handoff']
METHODS += ['original_perception_rim_position_body_brake_noangular_handoff',
            'original_perception_rim_position_track_brake_noangular_handoff']
SOURCES=['experiments/handoff_pose_gate.py','experiments/run_known_grasp_trial.py','experiments/lastinch_visual.py','experiments/cable_boundary.py',
    'experiments/diagnostic_json.py','experiments/run_tactile_registration.py','experiments/run_tactile_registration.sh',
    'experiments/run_force_tactile_registration.py','experiments/run_force_tactile_registration.sh',
    'experiments/tactile_geometry.py','experiments/wrench_noise.py','experiments/bootstrap_kilted_controller.py','experiments/cable_initialization.py',
    'experiments/run_seeded_evaluation.py','experiments/recording_tail.py','experiments/lastinch_geometry.py','experiments/headless_assets.py',
    'experiments/audit_sc_collision_fit.py','experiments/verification.py','experiments/resume_kilted.py',
    'aic_model/aic_model/insertion/contract.py',
    'experiments/run_physics_convergence.py','experiments/configs/numerical_development_study.json',
    'experiments/configs/matched_registration_development.json',
    'experiments/configs/bullet_double_binary_audit.json','experiments/configs/bullet_sharp_binary_audit.json','experiments/configs/bullet_damped_binary_audit.json','experiments/audit_physics_process.py',
    'experiments/physics_probe/double_backend_CMakeLists.txt','experiments/physics_probe/sharp_backend_CMakeLists.txt',
    'experiments/configs/bullet_sharp_source_provenance.json','experiments/configs/bullet_damped_source_provenance.json',
    'experiments/physics_probe/damped_backend_CMakeLists.txt',
    'experiments/physics_probe/CMakeLists.txt','experiments/physics_probe/solver_probe.cc',
    'experiments/contact_depth.py','experiments/runtime_contact_contract.py',
    'experiments/contact_contract.py','experiments/qualify_sfp_contact.py','experiments/analyze_recorded_wrench.py',
    'experiments/start_kilted_handoff.sh','experiments/kilted_handoff.launch.py','experiments/check_kilted_hold.py',
    'experiments/check_kilted_hold.sh','experiments/run_known_grasp_trial.sh','experiments/record_insertion.sh',
    'experiments/gazebo/ContactAudit.cc','experiments/gazebo/ResearchGrasp.cc',
    'experiments/prepare_kilted_handoff.py','experiments/lastinch_scene.py','experiments/record_insertion.py',
    'experiments/verify_recorded_trial.py','experiments/configs/sc_seating_profile.json',
    'experiments/configs/sfp_seating_profile.json','aic_model/aic_model/insertion/sc_visual_alignment.py']
SOURCES += ['experiments/run_original_insertion.py','experiments/run_original_insertion.sh','experiments/challenge_grasp.py','aic_model/aic_model/policy.py']
SOURCES += sorted(str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'aic_model/aic_model/insertion').glob('*.py')
                  if str(p.relative_to(ROOT)).replace('\\','/') not in SOURCES)
SOURCES += ['aic_example_policies/aic_example_policies/ros/perception_core.py']
SOURCES += ['aic_model/aic_model/insertion/sc_front_rim.npz','experiments/configs/sc_front_rim.json','aic_model/setup.py']
SOURCES += ['aic_model/aic_model/insertion/sc_body_template.npz','experiments/configs/sc_body_template.json','experiments/build_sc_body_template.py']
SOURCES += ['experiments/configs/visual_tracking_development.json']
SOURCES += ['experiments/configs/angular_screen_ablation30.json']
SOURCES += ['experiments/configs/component_ablation30.json']
SOURCES += ['flowstate/aic_perception/aic_perception/gripper_masks.py','flowstate/aic_perception/aic_perception/config.py']
SOURCES += ['aic_example_policies/aic_example_policies/ros/weights/'+n for n in
            ('best.pt','best_sc_mouth_pose.pt','best_sc_plug_pose.pt','best_sfp_plug_pose.pt')]

def validate_cases(cases):
    """Reject malformed targets before launching any simulator."""
    seen=set()
    for case in cases:
        connector=case['connector'];index=case['target_index'];port=case['sfp_port']
        if connector not in ('sc','sfp') or index not in range(5):raise ValueError(f'Invalid target: {case}')
        if port not in ((0,) if connector=='sc' else (0,1)):raise ValueError(f'Invalid {connector} slot: {port}')
        if case['case_id'] in seen:raise ValueError('Repeated case identity')
        seen.add(case['case_id'])
    return cases

def freeze():
    path=ROOT/'experiments/configs/development_cases.json'
    if path.exists():return path
    rng=np.random.default_rng(3000);cases=[]
    for i in range(10):
        cases.append({'case_id':f'dev_{3000+i}','seed':3000+i,'connector':'sc' if i%2==0 else 'sfp',
            'target_index':i//2,'sfp_port':(i//2)%2 if i%2 else 0,'gap_mm':float(rng.uniform(5,15)),
            'lateral_mm':rng.uniform(-.25,.25,2).tolist(),'angular_deg':rng.uniform(-.15,.15,3).tolist()})
    path.write_text(json.dumps({'purpose':'development_only_not_final','rng_seed':3000,'methods':METHODS,
        'scene_and_simulator_seed_shared':True,'camera_rate_hz':5,'remote_endpoint_support':False,'physics_step_s':.001,
        'cases':cases},indent=2)+'\n');return path

def cmd(args,log=None,timeout=350):
    if log:
        with open(log,'wb') as f:return subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=timeout,check=True)
    return subprocess.run(args,cwd=ROOT,capture_output=True,text=True,timeout=timeout,check=True)

def archive(label):
    source=(ART/'kilted_handoff').resolve();destination=(ART/'lastinch_runs'/label).resolve()
    workspace=ART.resolve()
    if workspace not in source.parents or workspace not in destination.parents:raise ValueError('Archive outside workspace')
    if source.exists():
        if destination.exists():raise FileExistsError(destination)
        destination.parent.mkdir(exist_ok=True);source.rename(destination)
    return destination

def trial(case,method,batch):
    for source,digest in batch['source_sha256'].items():
        if hashlib.sha256((ROOT/source).read_bytes()).hexdigest()!=digest:raise RuntimeError(f'Frozen source changed: {source}')
    run=f"{case['case_id']}_{method}";logs=ART/'container';logs.mkdir(exist_ok=True)
    server=None;recorder=None
    try:
        cmd(['docker','start',CONTAINER])
        scene=f"/research/.artifacts/lastinch_scenes/{case['case_id']}"
        args=['docker','exec','-w','/research',CONTAINER,'python3','-m','experiments.lastinch_scene','--out',scene,
            '--connector',case['connector'],'--target-index',str(case['target_index']),'--sfp-port',str(case['sfp_port']),
            '--gap-mm',str(case['gap_mm']),'--seed',str(case['seed']),'--native-dynamic','--lateral-mm',*map(str,case['lateral_mm']),
            '--angular-deg',*map(str,case['angular_deg']),'--cable-initialization',batch['cable_initialization'],
            '--grasp-profile',batch.get('grasp_profile','legacy_sfp')]
        if case.get('grasp_jitter_seed') is not None:
            args += ['--grasp-jitter-seed',str(case['grasp_jitter_seed'])]
        cmd(args,logs/f'{run}_scene.log')
        root=ART/'kilted_handoff';root.mkdir(parents=True,exist_ok=True)
        for source in SOURCES:
            destination=root/'source_snapshot'/source;destination.parent.mkdir(parents=True,exist_ok=True)
            if Path(source).suffix=='.pt':
                frozen=ART/'frozen_checkpoint_objects'/(batch['source_sha256'][source]+'.pt')
                if frozen.is_file() and hashlib.sha256(frozen.read_bytes()).hexdigest()==batch['source_sha256'][source]:
                    destination.hardlink_to(frozen)
                    continue
            shutil.copy2(ROOT/source,destination)
        with open(logs/f'{run}_server.log','wb') as f:
            server=subprocess.Popen(['docker','exec','-e',f'AIC_RESEARCH_SCENE={scene}','-e','AIC_RESEARCH_CAMERA_RATE=5',
                '-e',f"AIC_RESEARCH_REMOTE_SUPPORT={batch['remote_support']}",'-e',f"AIC_RESEARCH_PHYSICS_STEP={batch['physics_step']}",
                '-e',f"AIC_RESEARCH_PHYSICS_BACKEND={batch.get('physics_backend','pinned_float32')}",
                '-e',f"AIC_RESEARCH_WRIST_RATE={batch.get('wrist_rate',100)}",
                *(['-e',f"AIC_RESEARCH_SOLVER_ITERATIONS={batch['solver_iterations_requested']}"] if batch.get('solver_iterations_requested') is not None else []),
                CONTAINER,'bash','/research/experiments/start_kilted_handoff.sh'],stdout=f,stderr=subprocess.STDOUT)
        root=ART/'kilted_handoff';deadline=time.monotonic()+210
        while time.monotonic()<deadline:
            try:
                boot=json.loads((root/'controller_bootstrap.json').read_text())
                if boot.get('error'):raise RuntimeError(f"Controller bootstrap failed: {boot['error']}")
                if boot.get('ready'):break
            except (FileNotFoundError,json.JSONDecodeError):pass
            if server.poll() is not None:raise RuntimeError('Simulator exited')
            time.sleep(1)
        else:raise TimeoutError('Bootstrap')
        cmd(['docker','exec',CONTAINER,'python3','/research/experiments/audit_physics_process.py'],logs/f'{run}_physics_process.log')
        cmd(['docker','exec',CONTAINER,'bash','/research/experiments/check_kilted_hold.sh','--calibrate-load','--calibrate-reset'],logs/f'{run}_hold.log')
        hold=json.loads((root/'controller_hold_calibrated.json').read_text())
        if not hold['gate_passed']:raise RuntimeError('Hold validation failed')
        # Evaluator-owned prospective protocol, never inferred from policy success.
        scene_spec=json.loads((root/'scene.json').read_text())
        scene_spec['evaluation_force_ceiling_N']=batch.get('force_ceiling_N',8.)
        scene_spec['sfp_motion_profile']=batch.get('sfp_motion_profile','legacy36')
        (root/'scene.json').write_text(json.dumps(scene_spec,indent=2)+'\n')
        proof=Path(batch.get('positive_control_proof_file',ART/'sfp_contact_positive_control.json'))
        if proof.exists() and hashlib.sha256(proof.read_bytes()).hexdigest()!=batch.get('positive_control_proof_sha256'):
            raise RuntimeError('Frozen positive control proof changed')
        if proof.exists():shutil.copy2(proof,root/'sfp_contact_positive_control.json')
        env=['-e',f"AIC_RESEARCH_ABLATION={batch.get('ablation','full')}",'-e',f'AIC_RESEARCH_RUN_ID={run}','-e',f"AIC_RESEARCH_CONNECTOR={case['connector']}",
            '-e',f"AIC_RESEARCH_FORCE_CEILING_N={batch.get('force_ceiling_N',8.)}",
            '-e',f"AIC_SFP_MOTION_PROFILE={batch.get('sfp_motion_profile','legacy36')}",
            '-e',f"AIC_RESEARCH_GAP_MM={case['gap_mm']}",'-e',f'AIC_RESEARCH_METHOD={method}',
            '-e',f"AIC_RESEARCH_TARGET_INDEX={case['target_index']}",'-e',f"AIC_RESEARCH_SFP_PORT={case['sfp_port']}"]
        if method.startswith('original_perception'):
            env += ['-e','AIC_PLUG_POSE_REFINE_STRATEGY='+('geometric_selective' if method.endswith('_selective') else 'fixed')]
            for variable,source in batch.get('weight_overrides',{}).items():
                env += ['-e',variable+'=/research/'+source]
            if method.endswith('_reacquire') or '_rim' in method:
                env += ['-e','AIC_INSERT_VISUAL_REACQUIRE=1','-e','RL_INSERT_SC_STRICT_PERCEPTION=1']
            if '_rim' in method:
                env += ['-e','AIC_SC_RIM_REFINE=1','-e','AIC_SC_RIM_RASTER_OFFSET_PX=-0.5',
                        '-e','RL_INSERT_SC_MAX_SEAT_ATTEMPTS='+('2' if '_rim_contact' in method else '1')]
            if '_rim_contact' in method:
                env += ['-e','AIC_SC_CONTACT_REGISTER=1']
            if '_contact_search' in method:
                env += ['-e','AIC_SC_CONTACT_CAPTURE_SEARCH=1']
            if '_rim_position' in method:
                env += ['-e','AIC_INSERT_NATIVE_WRENCH_GAIN=0']
            if '_position_center' in method:
                env += ['-e','AIC_SC_TERMINAL_CONTACT_REGISTER=1']
            if '_position_body' in method:
                env += ['-e','AIC_SC_BODY_TRANSLATION=1']
            if '_body_track' in method or '_position_track' in method:
                env += ['-e','AIC_SC_LATERAL_TRACKING=1']
            if '_noangular_' in method:
                env += ['-e','AIC_SC_BODY_ANGULAR_SCREEN_ABLATION=1']
            if '_brake_' in method:
                env += ['-e','RL_INSERT_SC_TERMINAL_BRAKE=1']
            if '_axial_' in method:
                env += ['-e','AIC_SC_AXIAL_CONTACT_REGISTER=1','-e','AIC_SC_BODY_FRESH_RETRY=1']
            if '_axial_partial_' in method:
                env += ['-e','AIC_SC_PARTIAL_BODY_CONTACT=1']
            if method=='original_perception_sfp_geometry':
                env += ['-e','AIC_SFP_FULL_PORT_ORIENTATION=1','-e','AIC_INSERT_NATIVE_WRENCH_GAIN=0']
            if method=='original_perception_sfp_contact_datum_handoff':
                env += ['-e','AIC_SFP_CONTACT_DATUM=1','-e','AIC_INSERT_NATIVE_WRENCH_GAIN=0']
            if method.endswith('_handoff'):
                env += ['-e','AIC_SC_KEEP_HANDOFF_ORIENTATION=1']
        with open(logs/f'{run}_recorder.log','wb') as f:
            recorder=subprocess.Popen(['docker','exec',*env,CONTAINER,'bash','/research/experiments/record_insertion.sh'],stdout=f,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+15
        while not (root/run).exists() and time.monotonic()<deadline:time.sleep(.1)
        # Simple shell program, no interpolated command text or data.
        policy_script=('run_original_insertion.sh' if method.startswith('original_perception') else
            'run_force_tactile_registration.sh' if method.startswith('force_tactile_') else
            'run_tactile_registration.sh' if method.startswith('tactile_') else 'run_known_grasp_trial.sh')
        cmd(['docker','exec',*env,CONTAINER,'bash',f'/research/experiments/{policy_script}'],logs/f'{run}_policy.log',timeout=1550)
        recorder.wait(timeout=60)
        if recorder.returncode:raise RuntimeError(f'Recorder exit {recorder.returncode}')
        cmd([sys.executable,str(ROOT/'experiments/analyze_recorded_wrench.py'),'--run-dir',str(root/run)],logs/f'{run}_wrench.log')
        cmd([sys.executable,str(ROOT/'experiments/verify_recorded_trial.py'),'--scene-dir',str(root),'--run',run],logs/f'{run}_verify.log')
        result=json.loads((root/run/'verified.json').read_text());result['status']='completed'
        result['peak_independent_tared_force_N']=json.loads((root/run/'wrench_analysis.json').read_text())['peak_independent_tared_force_N']
    except Exception as exc:
        result={'run':run,'status':'environment_or_execution_failure','error':f'{type(exc).__name__}: {exc}'}
    finally:
        cmd(['docker','stop',CONTAINER],timeout=30)
        if server is not None:server.wait(timeout=15)
        if recorder is not None:recorder.wait(timeout=15)
        folder=archive(run)
    result.update(case=case,method=method,evidence_dir=str(folder.relative_to(ROOT)))
    batch['results'].append(result)
    (ART/'seeded_evaluation_progress.json').write_text(json.dumps(batch,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('run','status','verified_success','reason','error') if k in result}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--freeze-only',action='store_true')
    p.add_argument('--limit-cases',type=int,default=4);p.add_argument('--case-offset',type=int,default=0)
    p.add_argument('--methods',nargs='+',choices=METHODS,default=DEFAULT_METHODS)
    p.add_argument('--method-order-seed',type=int,help='Deterministic within-case method-order shuffle')
    p.add_argument('--case-ids',nargs='+',help='Explicit development identities; no final-manifest access')
    p.add_argument('--case-manifest',type=Path,help='Explicit development-only manifest, never a final/test split')
    p.add_argument('--nominal',choices=['sc','sfp','sfp_positive_control']);p.add_argument('--label',default='')
    p.add_argument('--remote-support',choices=['0','1'],default='0')
    p.add_argument('--grasp-profile',choices=['legacy_sfp','challenge_sc'],default='legacy_sfp')
    p.add_argument('--sc-mouth-weights',type=Path)
    p.add_argument('--sc-plug-weights',type=Path)
    p.add_argument('--nic-weights',type=Path)
    p.add_argument('--sfp-plug-weights',type=Path)
    p.add_argument('--physics-step',type=float,default=.002)
    p.add_argument('--wrist-rate',type=float,default=100)
    p.add_argument('--cable-initialization',choices=['source','sc_outboard','sc_outboard_horizontal','sc_straight_outboard'],default='source')
    p.add_argument('--solver-iterations',type=int)
    p.add_argument('--force-ceiling-N',type=float,choices=[8.,20.],default=8.)
    p.add_argument('--sfp-motion-profile',choices=['legacy36','fast36','fast37','fast38'],default='legacy36')
    p.add_argument('--physics-backend',choices=['pinned_float32','source_float64','source_float64_sharp','source_float64_damped'],default='pinned_float32')
    p.add_argument('--positive-control-proof',type=Path,default=ART/'sfp_contact_positive_control.json')
    a=p.parse_args();path=a.case_manifest or freeze()
    if a.sfp_motion_profile in ('fast36','fast37','fast38') and (a.force_ceiling_N!=20. or a.methods!=['original_perception_sfp_contact_datum_handoff']):
        raise ValueError('Fast profile requires the separately labelled 20 N SFP datum protocol')
    if not 0 <= a.wrist_rate <= min(1000,1/a.physics_step)+1e-6:raise ValueError('Wrist rate must fit physics step and legacy scheduler; use zero for every physics step')
    if a.solver_iterations is not None and a.solver_iterations<=0:raise ValueError('Positive solver iterations required')
    if a.freeze_only:print(path);return
    config=json.loads(path.read_text());all_cases=validate_cases(config['cases'])
    if a.case_manifest:
        if not str(config.get('purpose','')).startswith('development_only'):
            raise ValueError('Only an explicitly development-only manifest is accepted')
        if any(not c['case_id'].startswith('dev_') or not 0 <= int(c['seed']) < 100000 for c in all_cases):
            raise ValueError('Custom manifests must use development identities and seeds below 100000')
        source=str(path.resolve().relative_to(ROOT)).replace('\\','/')
        if source not in SOURCES:SOURCES.append(source)
    if a.case_ids:
        known={c['case_id']:c for c in all_cases}
        cases=[known[name] for name in a.case_ids]
    else:cases=all_cases[a.case_offset:a.case_offset+a.limit_cases]
    if a.nominal:
        connector='sc' if a.nominal=='sc' else 'sfp'
        cases=[{'case_id':f'{a.nominal}_supported','seed':1001,'connector':connector,'target_index':2,'sfp_port':0,
            'gap_mm':10,'lateral_mm':[2.,0.] if a.nominal=='sfp_positive_control' else [0.,0.],'angular_deg':[0.,0.,0.]}]
    if a.label:
        if not a.label.replace('_','').isalnum():raise ValueError('Invalid label')
        for case in cases:case['case_id']+='_'+a.label
    if a.sfp_motion_profile in ('fast36','fast37','fast38') and any(c['connector']!='sfp' for c in cases):
        raise ValueError('Fast SFP profile cannot be applied to SC cases')
    overrides={}
    for variable,weight_path in [('AIC_SC_POSE_WEIGHTS',a.sc_mouth_weights),('AIC_SC_PLUG_POSE_WEIGHTS',a.sc_plug_weights),
                                 ('AIC_NIC_WEIGHTS',a.nic_weights),('AIC_SFP_PLUG_POSE_WEIGHTS',a.sfp_plug_weights)]:
        if weight_path is None:continue
        if not weight_path.is_file():raise FileNotFoundError(weight_path)
        kind='sfp' if variable in ('AIC_NIC_WEIGHTS','AIC_SFP_PLUG_POSE_WEIGHTS') else 'sc'
        if any(not m.startswith('original_perception') for m in a.methods) or any(c['connector']!=kind for c in cases):
            raise ValueError(f'{kind} checkpoint overrides require matching original-perception methods')
        source=str(weight_path.resolve().relative_to(ROOT)).replace('\\','/')
        overrides[variable]=source
        if source not in SOURCES:SOURCES.append(source)
    batch={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'method_order_seed':a.method_order_seed,'method_order_by_case':{},
        'remote_support':a.remote_support,'physics_step':a.physics_step,'cable_initialization':a.cable_initialization,'grasp_profile':a.grasp_profile,
        'wrist_rate':a.wrist_rate,
        'force_ceiling_N':a.force_ceiling_N,'sfp_motion_profile':a.sfp_motion_profile,
        'solver_iterations_requested':a.solver_iterations,
        'physics_backend':a.physics_backend,
        'weight_overrides':overrides,
        'positive_control_proof_file':str(a.positive_control_proof.resolve()),
        'positive_control_proof_sha256':hashlib.sha256(a.positive_control_proof.read_bytes()).hexdigest() if a.positive_control_proof.exists() else None,
        'source_sha256':{source:hashlib.sha256((ROOT/source).read_bytes()).hexdigest() for source in SOURCES},
        'purpose':'development_only_final_100000_series_untouched','results':[]}
    cmd(['docker','stop',CONTAINER],timeout=30)
    archive('pre_batch_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    order_rng=np.random.default_rng(a.method_order_seed) if a.method_order_seed is not None else None
    for case in cases:
        order=list(a.methods)
        if order_rng is not None:order_rng.shuffle(order)
        batch['method_order_by_case'][case['case_id']]=order
        for method in order:trial(case,method,batch)
    (ART/('evaluation_'+(a.nominal or 'seeded')+'.json')).write_text(json.dumps(batch,indent=2)+'\n')

if __name__=='__main__':
    # Per-worker idle-sleep protection; no display or global power changes.
    power=None
    if sys.platform=='win32':
        import ctypes
        power=ctypes.windll.kernel32
        if not power.SetThreadExecutionState(0x80000001):
            raise RuntimeError('Could not request active evaluation idle-sleep protection')
    try:main()
    finally:
        if power is not None:power.SetThreadExecutionState(0x80000000)


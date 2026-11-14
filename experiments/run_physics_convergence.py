"""Fixed development-only numerical study; no policy tuning or final seeds.

Same seed 3003 and translation policy: engine-default/1ms, 200 iterations/1ms,
200 iterations/0.5ms. Each changed configuration needs its own executed SFP
collision-positive control. Repeated cases are never pooled as independent n.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.run_seeded_evaluation import SOURCES

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--label',required=True)
    p.add_argument('--contexts',nargs='+',choices=['default_dt1','iters200_dt1','iters200_dt05'],default=['default_dt1','iters200_dt1','iters200_dt05'])
    a=p.parse_args()
    if not a.label.replace('_','').isalnum():raise ValueError('Alphanumeric label required')
    out=ROOT/'.artifacts/physics_convergence'/a.label;out.mkdir(parents=True)
    state={'purpose':'fixed_development_numerical_diagnostic_final_seeds_untouched','pid':os.getpid(),
        'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'running','current_stage':None,'selected_contexts':a.contexts,
        'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES},'stages':[]}
    marker=ROOT/'.artifacts/active_physics_convergence.json'
    def save():
        (out/'progress.json').write_text(json.dumps(state,indent=2)+'\n')
        marker.write_text(json.dumps({'status':state['status'],'pid':state['pid'],'current_stage':state['current_stage'],
            'progress':str(out/'progress.json')},indent=2)+'\n')
    def command(args,log):
        for name,digest in state['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError(f'Frozen source changed: {name}')
        with (out/log).open('wb') as stream:subprocess.run(args,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=2400)
    def execute(name,step,iters,control=False,proof=None):
        label=a.label+'_'+name;state['current_stage']=name;save()
        args=[sys.executable,'experiments/run_seeded_evaluation.py','--label',label,'--physics-step',str(step),'--remote-support','0']
        args+=['--nominal','sfp_positive_control','--methods','initial_pose_guarded'] if control else ['--case-ids','dev_3003','--methods','force_tactile_translation']
        if iters is not None:args+=['--solver-iterations',str(iters)]
        if proof:args+=['--positive-control-proof',str(proof)]
        command(args,name+'.log')
        source=ROOT/'.artifacts'/('evaluation_sfp_positive_control.json' if control else 'evaluation_seeded.json')
        data=json.loads(source.read_text());data['development_revision']='numerics_'+name
        data['numerical_configuration']={'timestep_s':step,'solver_iterations_requested':iters}
        index=out/(name+'.json');index.write_text(json.dumps(data,indent=2)+'\n')
        result=data['results'][0];state['stages'].append({'name':name,'index':str(index.relative_to(ROOT)),
            'status':result['status'],'reason':result.get('reason',result.get('error')),
            'verified_success':result.get('verified_success'),'max_global_penetration_m':result.get('max_global_penetration_m'),
            'peak_independent_tared_force_N':result.get('peak_independent_tared_force_N')});save()
        print(json.dumps(state['stages'][-1]),flush=True)
        return result
    try:
        proof=ROOT/'.artifacts/sfp_contact_positive_control.json';existing=json.loads(proof.read_text())
        if not existing.get('qualified') or existing.get('timestep_s')!=.001 or existing.get('solver_iterations_requested') is not None:
            raise ValueError('Existing default/1ms collision control is not qualified')
        state['current_stage']='solver_capability_readback';save()
        command(['docker','start','research-aic-eval'],'solver_probe_start.log')
        command(['docker','exec','research-aic-eval','bash','-c',
            'source /ws_aic/install/setup.bash && exec /opt/research-solver-probe/solver_probe /ws_aic/install/lib/libgz-physics8-bullet-featherstone-plugin.so.8.3.0'],
            'solver_probe.log')
        readback=json.loads((out/'solver_probe.log').read_text())
        if readback['default_iterations_readback']!=50 or readback['iterations_readback']!=200:raise RuntimeError('Unexpected solver readback')
        state['solver_capability_readback']=readback;save()
        for name,step in [('iters200_dt1',.001),('iters200_dt05',.0005)]:
            if name not in a.contexts:continue
            result=execute(name+'_control',step,200,control=True)
            if result['status']!='completed':
                state['stages'].append({'name':name,'status':'not_executed_unqualified_contact_control'});save();continue
            qualified=out/(name+'_contact_proof.json')
            try:
                command([sys.executable,'experiments/qualify_sfp_contact.py','--scene-dir',str(ROOT/result['evidence_dir']),
                    '--run',result['run'],'--out',str(qualified)],name+'_qualify.log')
            except subprocess.CalledProcessError:
                state['stages'].append({'name':name,'status':'not_executed_unqualified_contact_control'});save();continue
            execute(name,step,200,proof=qualified)
        if 'default_dt1' in a.contexts:execute('default_dt1',.001,None,proof=proof)
        state['status']='complete';state['current_stage']=None
    except Exception as exc:
        state['status']='failed';state['error']=f'{type(exc).__name__}: {exc}'
        subprocess.run(['docker','stop','research-aic-eval'],cwd=ROOT,capture_output=True,timeout=30)
        raise
    finally:
        state['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()

if __name__=='__main__':main()

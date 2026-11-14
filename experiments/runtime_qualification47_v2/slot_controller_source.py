"""Run known development cases between unchanged final44 trials, one world only.

Suspend only the owned coordinator at a stopped-container boundary. A separate
watchdog releases that suspension if this process loses its heartbeat. Neither
the final manifest nor its sources, ledger, policy or outcomes are edited.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import psutil

ROOT=Path(__file__).resolve().parents[1]
DEV=ROOT/'.artifacts/dev46_workspace'
OUT=ROOT/'experiments/runtime_qualification47'
CONTAINER='research-aic-perception30'
REVISION=1


def write(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def owned(lease):
    process=psutil.Process(lease['pid'])
    if abs(process.create_time()-lease['created'])>.01 or not any('resume_full_evaluation44.py' in s for s in process.cmdline()):
        raise RuntimeError('Coordinator process identity changed')
    return process


def docker_running(name):
    return subprocess.run(['docker','inspect',name,'--format','{{.State.Running}}'],capture_output=True,text=True,
                          check=True,timeout=20).stdout.strip()=='true'


def stop_development():
    subprocess.run(['docker','stop','--timeout','5',CONTAINER],capture_output=True,timeout=40,check=True)
    if docker_running(CONTAINER):raise RuntimeError('Development container still running; cannot resume second world')


def watchdog(path):
    while True:
        lease=read(path)
        if lease['state'] in ('released','aborted_before_slot'):return
        if lease['state'] in ('suspend_intent','active') and time.time()-lease['heartbeat']>75:
            stop_development();owned(lease).resume()
            write(path.with_name('watchdog_recovery.json'),dict(recovered_unix=time.time(),lease=lease))
            return
        time.sleep(5)


def main():
    OUT.mkdir(exist_ok=True)
    lease_path=OUT/'slot_lease.json'
    if lease_path.exists():raise FileExistsError('Preserve previous slot; inspect before another version')
    processes=[p for p in psutil.process_iter(['pid','cmdline','create_time'])
               if p.pid!=os.getpid() and p.info['cmdline'] and
               any('resume_full_evaluation44.py' in s for s in p.info['cmdline'])]
    if len(processes)!=1:raise RuntimeError('Expected one owned final coordinator')
    process=processes[0]
    lease=dict(pid=process.pid,created=process.create_time(),state='waiting_for_completed_trial',heartbeat=time.time(),
               development_container=CONTAINER,final_policy_changes=False)
    write(lease_path,lease)
    guard=subprocess.Popen([sys.executable,str(Path(__file__)),'--watchdog',str(lease_path)],cwd=ROOT,
                           creationflags=subprocess.CREATE_NO_WINDOW)
    lock=threading.Lock();end=threading.Event();suspended=False
    def heartbeat():
        while not end.wait(10):
            with lock:lease['heartbeat']=time.time();write(lease_path,lease)
    threading.Thread(target=heartbeat,daemon=True).start()
    def state(value,**extra):
        with lock:lease.update(state=value,heartbeat=time.time(),**extra);write(lease_path,lease)
    freeze=read(ROOT/'experiments/full_evaluation44/source_freeze.json')['source_sha256']
    def unchanged():
        if any(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h for p,h in freeze.items()):
            raise RuntimeError('Frozen final source changed')
    report=dict(scope='Known development runtime qualification, no final cases used',results=[],status='waiting_for_slot')
    def save():write(OUT/'progress.json',report)
    save()
    try:
        deadline=time.monotonic()+1500
        while time.monotonic()<deadline:
            final=read(ROOT/'experiments/full_evaluation44/progress.json')
            if final.get('current_run') is None and final['status']=='primary_final_trials':
                state('suspend_intent');owned(lease).suspend();suspended=True
                # Any already-started Docker command must finish before the
                # second check. Resuming immediately on a race preserves order.
                time.sleep(2)
                current=read(ROOT/'experiments/full_evaluation44/progress.json')
                if current.get('current_run') is None and not docker_running('research-aic-eval'):
                    unchanged();state('active',final_completed_before_slot=len(current['results']));break
                owned(lease).resume();suspended=False;state('waiting_for_completed_trial')
            if not process.is_running():raise RuntimeError('Final coordinator exited before slot')
            time.sleep(.15)
        else:raise TimeoutError('No safe final boundary observed; no development world started')
        report.update(status='runtime_qualification',final_completed_before_slot=lease['final_completed_before_slot']);save()
        subprocess.run(['docker','update','--cpus','8','--memory','12g','--memory-swap','13g',CONTAINER],
                       check=True,capture_output=True,timeout=30)
        for kind,case,manifest,method in [
            ('sc','dev_42000','sc_finish41_qualification.json','original_perception_rim_position_body_track_brake_axial_partial_noangular_handoff'),
            ('sfp','dev_45005','sfp_contact34_development.json','original_perception_sfp_contact_datum_handoff')]:
            unchanged();report['current']=kind;save()
            args=[sys.executable,'experiments/run_seeded_evaluation.py','--case-manifest','experiments/configs/'+manifest,
                  '--case-ids',case,'--methods',method,'--label',f'runtime47v{REVISION}','--physics-backend','source_float64_damped',
                  '--physics-step','.0005','--wrist-rate','0','--solver-iterations','200','--force-ceiling-N','20',
                  '--positive-control-proof','experiments/full_evaluation44/contact_positive_control.json']
            weights='experiments/recovery_20261112/checkpoints/'
            if kind=='sc':args+=['--grasp-profile','challenge_sc','--cable-initialization','sc_straight_outboard',
                                '--sc-mouth-weights',weights+'sc_mouth_recovery20261112/best.pt']
            else:args+=['--sfp-motion-profile','fast38','--nic-weights',weights+'nic_recovery20261112/best.pt',
                       '--sfp-plug-weights',weights+'plug_recovery20261112/best.pt']
            with (OUT/(kind+'_executor.log')).open('wb') as log:
                subprocess.run(args,cwd=DEV,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1500)
            batch=read(DEV/'.artifacts/evaluation_seeded.json');result=batch['results'][0]
            report['results'].append(result);save()
            if not result.get('verified_success'):raise RuntimeError('Development case did not independently verify; preserve and inspect')
        unchanged();report.update(status='complete_requires_review',current=None);save()
    except Exception as exc:
        report.update(status='stopped_requires_inspection',error=f'{type(exc).__name__}: {exc}');save();raise
    finally:
        if suspended:
            # Keep heartbeat alive through cleanup; watchdog also refuses to
            # resume the main process until development Gazebo is stopped.
            stop_development();unchanged();owned(lease).resume();suspended=False
            state('released',released_unix=time.time())
        else:state('aborted_before_slot')
        end.set();guard.wait(timeout=15)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watchdog',type=Path);parser.add_argument('--revision',type=int,default=1);args=parser.parse_args()
    REVISION=args.revision
    if REVISION!=1:OUT=ROOT/f'experiments/runtime_qualification47_v{REVISION}'
    if args.watchdog:watchdog(args.watchdog)
    else:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:main()
        finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

"""Finish finite lightweight development checks; never launches a simulator."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/parallel_dev46'


def main():
    status=dict(scope='Offline development only; no second Gazebo process',status='waiting_for_six_paired_image_cases')
    def save():
        status['updated_unix']=time.time();(OUT/'offline_progress.json').write_text(json.dumps(status,indent=2))
    save();deadline=time.monotonic()+1200
    try:
        while True:
            path=OUT/'cropped_device_benchmark.json'
            if path.exists():
                data=json.loads(path.read_text());status['paired_cases_completed']=len(data['rows']);save()
                if len(data['rows'])==6:break
            if time.monotonic()>deadline:raise TimeoutError('Finite image benchmark incomplete; inspect logs')
            time.sleep(5)
        status['status']='verifying_real_cpu_thread_callback';save()
        with (OUT/'thread_callback_stdout.log').open('wb') as log:
            subprocess.run(['docker','exec','research-aic-perception30','/opt/research-perception/bin/python',
                            '/research/experiments/verify_thread_candidate46.py'],stdout=log,stderr=subprocess.STDOUT,
                           check=True,timeout=180,cwd=ROOT)
        subprocess.run([sys.executable,'experiments/summarize_parallel_dev46.py'],cwd=ROOT,check=True,capture_output=True)
        status.update(status='completed',final_policy_modified=False,physical_insertion_claimed=False);save()
    except Exception as exc:
        status.update(status='stopped_requires_inspection',error=f'{type(exc).__name__}: {exc}');save();raise


if __name__=='__main__':main()

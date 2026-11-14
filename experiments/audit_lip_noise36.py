"""Negative sensor-control audit: clipped motion with noise but no reaction.

This ideal fixture deliberately disables physical reaction in the sensor.
It is an observability negative control, not a Gazebo outcome or real sensor.
"""
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('axial_contract',ROOT/'aic_model/aic_model/insertion/axial_contact_registration.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def single(seed,speed=.0004):
    rng=np.random.default_rng(seed);dt=.02;point=command=-.0015;samples=[]
    for step in range(1400):
        t=step*dt
        samples.append([t,point,command,float(rng.normal(0,.2))])
        samples=[r for r in samples if r[0]>=t-.9]
        result=module.blocked_window(samples)
        if result:return dict(accepted=True,sim_s=t,reaction_increase_N=result['reaction_increase_N'])
        command=min(command+speed*dt,point+.0003,.003)
        point=min(command,.0000535)
    return dict(accepted=False)


def main():
    rows=[dict(seed=7000+i,**single(7000+i)) for i in range(32)]
    result=dict(ideal_sensor_negative_control_only=True,simulator_trials=False,
        physical_sensor_reaction_disabled=True,noise_standard_deviation_N=.2,
        unchanged_reaction_threshold_N=.025,seeds=rows,
        false_acceptances=sum(r['accepted'] for r in rows),total=len(rows),
        interpretation='No calibrated confidence claim is justified by the fixed reaction-rise gate; prospective sensor-noise-aware validation is required.')
    out=ROOT/'.artifacts/repair36/lip_noise_negative_control.json'
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='seeds'},indent=2))


if __name__=='__main__':main()

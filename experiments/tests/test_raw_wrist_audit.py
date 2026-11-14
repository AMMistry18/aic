"""An independently measured short impulse must not disappear through resampling."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.analyze_recorded_wrench import audit_raw_force


def stream():
    return [dict(header=dict(stamp=dict(sec=0,nanosec=i*500000),frame_id='sensor'),
        wrench=dict(force=dict(x=1.,y=2.,z=3.))) for i in range(1000)]


def test_single_physics_step_impulse_fails_even_between_50hz_samples():
    data=stream();data[501]['wrench']['force']['z']+=20
    result=audit_raw_force(data,.1,.49,2000,.0005)
    assert result['complete_action_coverage']
    assert not result['qualified']
    assert result['reason']=='raw_force_exceeded_8N'
    assert result['samples_over_8N']==1
    assert result['peak_tared_force_N']==20


def test_complete_static_load_passes_and_one_missing_step_fails():
    data=stream()
    assert audit_raw_force(data,.1,.49,2000,.0005)['qualified']
    data.pop(501)
    result=audit_raw_force(data,.1,.49,2000,.0005)
    assert not result['qualified']
    assert result['reason']=='raw_force_coverage_gap'


def test_missing_start_or_nonfinite_sensor_cannot_pass():
    data=stream()
    assert not audit_raw_force(data[300:],.1,.49,2000,.0005)['qualified']
    data[501]['wrench']['force']['x']=np.nan
    assert not audit_raw_force(data,.1,.49,2000,.0005)['qualified']


def test_prospective_20N_profile_keeps_old8_and_full_rate_coverage():
    data=stream();data[501]['wrench']['force']['z']+=19.
    assert not audit_raw_force(data,.1,.49,2000,.0005)['qualified']
    result=audit_raw_force(data,.1,.49,2000,.0005,20.)
    assert result['qualified'] and result['force_ceiling_N']==20.
    assert result['samples_over_8N']==1 and result['samples_over_ceiling']==0
    data[501]['wrench']['force']['z']+=2.
    assert audit_raw_force(data,.1,.49,2000,.0005,20.)['reason']=='raw_force_exceeded_20N'
    data.pop(501)
    assert not audit_raw_force(data,.1,.49,2000,.0005,20.)['qualified']

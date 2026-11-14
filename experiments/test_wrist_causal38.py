"""Clock/topic reordering controls: retain age limits and exclude future data."""
from types import SimpleNamespace as N
import pytest
from experiments.causal_wrist38_candidate import WristSensorCache

def message(t,frame='ati/tool_link'):
    return N(header=N(stamp=N(sec=int(t),nanosec=round((t-int(t))*1e9)),frame_id=frame))

def cache(now):
    clock=N(now=now)
    node=N(get_clock=lambda:N(now=lambda:N(nanoseconds=round(clock.now*1e9))))
    return WristSensorCache(node,subscribe=False),clock

def test_observed_15_5ms_clock_lag_selects_existing_past_sample():
    sensor,clock=cache(1.)
    for i in range(72):sensor.accept(message(.98+i*.0005))
    image=object();obs=N(left_image=image,wrist_wrench=message(.8))
    result=sensor.merge(obs)
    assert sensor.stamp(result.wrist_wrench)==pytest.approx(1.)
    assert result.left_image is image and sensor.stamp(obs.wrist_wrench)==.8
    assert sensor.diagnostics()['max_latest_ahead_of_clock_s']==pytest.approx(.0155)
    clock.now=1.01
    assert sensor.stamp(sensor.merge(obs).wrist_wrench)==pytest.approx(1.01)

def test_future_samples_are_never_used_even_inside_old_ten_ms_tolerance():
    sensor,_=cache(1.);sensor.accept(message(.9995));sensor.accept(message(1.005))
    assert sensor.stamp(sensor.merge(N()).wrist_wrench)==pytest.approx(.9995)

def test_actual_fifty_hz_policy_stream_with_15_5ms_clock_lead():
    sensor,_=cache(1.)
    for t in [.9755,.9955,1.0155]:sensor.accept(message(t))
    assert sensor.stamp(sensor.merge(N()).wrist_wrench)==pytest.approx(.9955)
    assert sensor.diagnostics()['max_policy_age_sim_s']==pytest.approx(.0045)

@pytest.mark.parametrize('stamps',[[1.001],[.899],[.8,1.015]])
def test_no_causal_fresh_sample_remains_a_rejection(stamps):
    sensor,_=cache(1.)
    for stamp in stamps:sensor.accept(message(stamp))
    with pytest.raises(RuntimeError):sensor.merge(N())

def test_duplicate_and_out_of_order_samples_cannot_change_selected_force():
    sensor,_=cache(1.);first=message(.999)
    for m in [first,message(.999),message(.998),message(1.015)]:sensor.accept(m)
    assert sensor.merge(N()).wrist_wrench is first
    assert sensor.unique==2

def test_bounded_history_does_not_bypass_freshness_or_frame_checks():
    sensor,_=cache(1.)
    for i in range(1000):sensor.accept(message(.5+i*.001))
    assert len(sensor._samples)==512
    sensor,_=cache(1.);sensor.accept(message(.999,'wrong_frame'))
    with pytest.raises(RuntimeError,match='frame'):sensor.merge(N())

def test_clock_catching_up_exposes_new_unique_measurements_without_relabeling():
    sensor,clock=cache(1.)
    for t in [.999,1.001,1.002]:sensor.accept(message(t))
    selected=[]
    for t in [1.,1.001,1.002]:
        clock.now=t;selected.append(sensor.stamp(sensor.merge(N()).wrist_wrench))
    assert selected==pytest.approx([.999,1.001,1.002])

def test_real_ros_observation_keeps_camera_exposure_and_original_wrench():
    Observation=pytest.importorskip('aic_model_interfaces.msg').Observation
    WrenchStamped=pytest.importorskip('geometry_msgs.msg').WrenchStamped
    sensor,_=cache(1.);obs=Observation();past=WrenchStamped();future=WrenchStamped()
    for sample,sec,nsec in [(past,0,999500000),(future,1,15500000)]:
        sample.header.frame_id='ati/tool_link';sample.header.stamp.sec=sec;sample.header.stamp.nanosec=nsec
        sensor.accept(sample)
    result=sensor.merge(obs)
    assert result is not obs and result.left_image is obs.left_image
    assert obs.wrist_wrench.header.stamp.nanosec==0
    assert result.wrist_wrench is past

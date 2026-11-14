from types import SimpleNamespace
import numpy as np
from aic_model.insertion.observable_terminal import SeatDwell
from aic_model.insertion.controller_clock import control_time


def test_seating_candidate_needs_continuous_full_second():
    dwell=SeatDwell(.01564,.00025,.25)
    assert not any(dwell.observe(i*.02,[0,0,.01564],.1,1.) for i in range(50))
    assert dwell.observe(1.,[0,0,.01564],.1,1.)


def test_seating_geometry_failure_and_stamp_gap_reset_dwell():
    dwell=SeatDwell(.0458,.0001,.1)
    for i in range(40):assert not dwell.observe(i*.02,[0,0,.0458],0.,0.)
    assert not dwell.observe(.8,[.0002,0,.0458],0.,0.)
    assert dwell.start is None
    assert not dwell.observe(2.,[0,0,.0458],0.,0.)
    assert not dwell.observe(3.,[0,0,.0458],0.,0.)
    assert dwell.start==3.


def test_force_and_invalid_values_cannot_certify_seating():
    for force in [8.01,float('nan')]:
        dwell=SeatDwell(.01564,.00025,.25)
        for i in range(100):assert not dwell.observe(i*.02,[0,0,.01564],0.,force)
        assert dwell.start is None


def test_polling_headroom_handles_command_overhead_without_relaxing_gap():
    # The former 20ms sleep plus 12ms lookup/command delay never qualified.
    slow=SeatDwell(.01564,.00025,.25)
    assert not any(slow.observe(i*.032,[0,0,.01564],0.,0.) for i in range(200))
    assert slow.gap_resets==199
    # 5ms sleep leaves headroom for that same 12ms delay, unchanged 30ms gate.
    fast=SeatDwell(.01564,.00025,.25)
    assert any(fast.observe(i*.017,[0,0,.01564],0.,0.) for i in range(70))
    assert fast.maximum_gap_s<.03


def test_physical_clock_is_independent_of_slow_wall_execution(monkeypatch):
    import aic_model.insertion.controller_clock as module
    node=SimpleNamespace(get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=2_500_000_000)))
    policy=SimpleNamespace(_parent_node=node)
    monkeypatch.setattr(module.time,'monotonic',lambda:900.)
    monkeypatch.setenv('AIC_INSERT_CONTROLLER_SIM_TIME','1')
    assert control_time(policy)==2.5
    monkeypatch.setenv('AIC_INSERT_CONTROLLER_SIM_TIME','0')
    assert control_time(policy)==900.

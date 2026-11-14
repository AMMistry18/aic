import numpy as np
from experiments.recording_tail import RecordingTail


def test_actual_terminal_gap_requires_more_evidence_instead_of_relaxed_dwell():
    tail=RecordingTail()
    # The recorded SFP failure's 1.68s gap cannot bridge terminal dwell.
    assert not tail.observe(68.74)
    for stamp in np.arange(70.42,71.42,.01):assert not tail.observe(float(stamp))
    assert tail.diagnostics()['observed_continuous_sim_s']<1.01
    complete=False
    for stamp in np.arange(71.42,72.1,.01):complete=tail.observe(float(stamp)) or complete
    assert complete and tail.diagnostics()['geometry_used'] is False


def test_repeated_clock_and_sparse_pose_messages_do_not_complete_tail():
    tail=RecordingTail()
    for _ in range(1000):assert not tail.observe(1.)
    for stamp in range(2,10):assert not tail.observe(float(stamp))

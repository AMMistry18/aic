import numpy as np
from scipy.spatial.transform import Rotation
from experiments.handoff_pose_gate import check_tip_window


def test_actual_tip_rejects_wrong_calibration_even_with_perfect_tcp_hold():
    desired = np.eye(4)
    displaced = desired.copy()
    displaced[:3, 3] = [.0068, 0, -.0111]
    times = np.linspace(.5, 1, 51)
    assert check_tip_window([(t, desired) for t in times], desired, 1)['passed']
    assert not check_tip_window([(t, displaced) for t in times], desired, 1)['passed']


def test_stale_or_twisted_handoff_rejected():
    desired = np.eye(4)
    twisted = desired.copy()
    twisted[:3, :3] = Rotation.from_euler('z', .5, degrees=True).as_matrix()
    assert not check_tip_window([(t, twisted) for t in np.linspace(.5, 1, 51)], desired, 1)['passed']
    assert not check_tip_window([(.5, desired), (1, desired)], desired, 1)['passed']
    assert not check_tip_window([], desired, 1)['passed']

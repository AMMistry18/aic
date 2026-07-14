from types import SimpleNamespace

import cv2
import numpy as np

from aic_model.board_search import BoardSearch, CENTER_TOL_FRAC, MAX_MOVES


def _frame(center=(320, 240), size=(220, 150), image_size=(640, 480)):
    width, height = image_size
    image = np.full((height, width, 3), 220, dtype=np.uint8)
    cx, cy = center
    bw, bh = size
    x0, y0 = round(cx - bw / 2), round(cy - bh / 2)
    x1, y1 = round(cx + bw / 2), round(cy + bh / 2)
    cv2.rectangle(image, (x0, y0), (x1, y1), (50, 50, 50), -1)
    # Bright saturated board marking must not prevent finding the outer plate.
    cv2.rectangle(image, (round(cx - 15), round(cy - 15)),
                  (round(cx + 15), round(cy + 15)), (255, 0, 255), -1)
    # A separate ragged dark shadow is intentionally less solid than the board.
    shadow = np.array([[5, 300], [170, 335], [30, 370], [150, 430], [5, 450]])
    cv2.fillPoly(image, [shadow], (75, 75, 75))
    return image


def test_detect_board_chooses_compact_plate():
    image = _frame(center=(390, 190))
    found, cx, cy, area_frac, bbox, touches, mask = BoardSearch(None).detect_board(image)
    assert found
    assert abs(cx - 390) < 2 and abs(cy - 190) < 2
    assert 0.07 < area_frac < 0.13
    assert bbox is not None and not touches
    assert mask.shape == image.shape[:2]


def test_detect_board_reports_clipped_plate():
    image = _frame(center=(50, 180))
    detection = BoardSearch(None).detect_board(image)
    assert detection[0]
    assert detection[5]


class _Logger:
    def info(self, _message):
        pass

    def error(self, _message):
        pass


class _FakePolicy:
    def __init__(self):
        self.position = np.array([0.0, 0.0, 0.50])
        self.moves = []
        self.jacobian = np.array([[-1800.0, 200.0], [100.0, -1800.0]])

    def _image(self):
        centroid = np.array([380.0, 190.0]) + self.jacobian @ self.position[:2]
        return _frame(center=centroid)

    def _get_cam_data(self, _obs, _cam_name):
        return self._image(), np.eye(3)

    def _tcp(self):
        return self.position.copy(), np.array([1.0, 0.0, 0.0, 0.0])

    def _lookup_cam_from_base(self, _cam_name):
        return np.eye(4)

    def set_pose_target(self, _move_robot, pose, frame_id="base_link"):
        assert frame_id == "base_link"
        self.position = np.array([pose.position.x, pose.position.y, pose.position.z])
        self.moves.append(self.position.copy())

    def sleep_for(self, _seconds):
        pass

    def get_logger(self):
        return _Logger()


def test_two_axis_probe_then_correction_centers_within_three_moves():
    policy = _FakePolicy()
    search = BoardSearch(policy)
    get_observation = lambda: SimpleNamespace()
    assert search.run(get_observation, lambda _update: None)
    assert len(policy.moves) == MAX_MOVES
    final = search.detect_board(policy._image())
    assert not final[5]
    assert abs(final[1] - 320) <= CENTER_TOL_FRAC * 640
    assert abs(final[2] - 240) <= CENTER_TOL_FRAC * 480


class _ViewingAxisFakePolicy(_FakePolicy):
    """Base X is optical depth, reproducing the singular v36 probe geometry."""

    def __init__(self):
        super().__init__()
        self.home = self.position.copy()
        self.jacobian = np.array([[0.0, 200.0, -1800.0], [0.0, -1800.0, 100.0]])

    def _image(self):
        centroid = np.array([380.0, 190.0]) + self.jacobian @ (self.position - self.home)
        return _frame(center=centroid)

    def _lookup_cam_from_base(self, _cam_name):
        # camera X -> base Z; camera Y -> base Y; camera Z -> base X
        T = np.eye(4)
        T[:3, :3] = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        return T


def test_camera_frame_probes_work_when_base_x_is_viewing_axis():
    policy = _ViewingAxisFakePolicy()
    search = BoardSearch(policy)
    assert search.run(lambda: SimpleNamespace(), lambda _update: None)
    assert len(policy.moves) == MAX_MOVES
    # The first probe is camera-image U, which is base Z in this pose, not base X.
    assert policy.moves[0][2] > policy.home[2]
    final = search.detect_board(policy._image())
    assert not final[5]
    assert abs(final[1] - 320) <= CENTER_TOL_FRAC * 640
    assert abs(final[2] - 240) <= CENTER_TOL_FRAC * 480

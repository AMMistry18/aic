"""Fail-closed multiview SC plug-pose estimation.

This is the SC sibling of :mod:`aic_model.sfp_plug_pose`.  It exists so the SC
insertion path can obtain ``sc_tip_link`` from vision on every run instead of
applying a fixed TCP->tip constant borrowed from the SFP plug.

The estimator is a *parameterisation* of :class:`SfpPlugPoseEstimator`, not a
fork: the triangulation, Kabsch fit, reprojection audit and rejection gates are
object-agnostic and already reviewed, so the only SC-specific inputs are the
eight local keypoints from :mod:`aic_model.sc_plug_pose_geometry` and a
separately trained YOLO-pose checkpoint.

The fail-closed contract of the SFP module is preserved exactly.  There is
deliberately **no fixed-grasp or bias fallback** anywhere in this module: a
missing checkpoint, stale frames, fewer than two usable cameras, or geometry
that does not fit the known plug returns ``None`` (or raises for malformed
configuration) so the caller stops rather than driving on a fabricated pose.
A constant fallback is also disallowed by the competition rules.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np

from .sc_plug_pose_geometry import SC_PLUG_LOCAL_KEYPOINTS_M
from .sfp_plug_pose import (
    PlugPoseEstimate,
    PlugPoseView,
    SfpPlugPoseEstimator,
)


#: Environment override for the trained SC plug-pose checkpoint.
SC_PLUG_POSE_WEIGHTS_ENV = "AIC_SC_PLUG_POSE_WEIGHTS"

#: Checkpoint written by ``tools/perception/sc_plug/train.py`` when no override is set.
SC_PLUG_POSE_WEIGHTS_BASENAME = "best_sc_plug_pose.pt"

# Rejection gates.  These start at the SFP values because the two plugs are of
# comparable size (the SC keypoint cuboid is 20.0 x 6.4 x 12.0 mm, the SFP one
# 12.8 x 7.6 x 18.0 mm), so the geometric conditioning of the fit is similar.
# They are rejection gates, not accuracy claims: a pose that survives them is
# only guaranteed to be self-consistent, not correct to any particular
# tolerance.  Re-tune them against measured held-out error before relying on
# them for seating -- tools/perception/sc_plug/validate.py reports the distributions needed
# to do that.
SC_MAX_REPROJECTION_ERROR_PX = 6.0
SC_MAX_KEYPOINT_RMSE_M = 0.0035


def default_sc_plug_pose_weights() -> Path | None:
    """Return the configured SC plug-pose checkpoint path, if one exists.

    Resolution order is the ``AIC_SC_PLUG_POSE_WEIGHTS`` environment override,
    then the in-repo weights directory shared with the SFP checkpoint.  A
    missing file yields ``None`` so the caller can fail closed with a clear
    message rather than constructing an estimator that cannot predict.
    """

    override = os.environ.get(SC_PLUG_POSE_WEIGHTS_ENV)
    if override:
        path = Path(override).expanduser()
        return path if path.is_file() else None

    # aic_model/aic_model/sc_plug_pose.py -> repo root is three parents up.
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root
        / "aic_example_policies"
        / "aic_example_policies"
        / "ros"
        / "weights"
        / SC_PLUG_POSE_WEIGHTS_BASENAME,
        # v50 overlay layout: docker/aic_model/v50_overlay/{aic_model,
        # aic_example_policies}, i.e. one aic_example_policies level, so the
        # overlay copy of this file resolves its sibling weights too.
        Path(__file__).resolve().parents[1]
        / "aic_example_policies"
        / "ros"
        / "weights"
        / SC_PLUG_POSE_WEIGHTS_BASENAME,
        Path(__file__).resolve().parent / "weights" / SC_PLUG_POSE_WEIGHTS_BASENAME,
        # Container convention, mirroring the SFP checkpoint at /models.
        Path("/models") / SC_PLUG_POSE_WEIGHTS_BASENAME,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


class ScPlugPoseEstimator(SfpPlugPoseEstimator):
    """YOLO-pose detector and strict multiview fuser for the SC duplex plug.

    Returns the world pose of ``sc_tip_link``.  Because
    ``SC_PLUG_LOCAL_KEYPOINTS_M`` is expressed in that frame, the fitted
    translation *is* the plug tip origin and the fitted rotation's third column
    is the insertion axis -- no additional offset is applied, and none may be.
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        *,
        max_reprojection_error_px: float = SC_MAX_REPROJECTION_ERROR_PX,
        max_keypoint_rmse_m: float = SC_MAX_KEYPOINT_RMSE_M,
        **kwargs,
    ):
        if "local_keypoints_m" in kwargs:
            raise TypeError(
                "ScPlugPoseEstimator fixes local_keypoints_m to the SC plug geometry"
            )
        super().__init__(
            weights_path,
            max_reprojection_error_px=max_reprojection_error_px,
            max_keypoint_rmse_m=max_keypoint_rmse_m,
            local_keypoints_m=SC_PLUG_LOCAL_KEYPOINTS_M,
            **kwargs,
        )

    def estimate_tip_pose(
        self,
        views: Sequence[PlugPoseView],
        *,
        now_s: float | None = None,
        max_age_s: float | None = None,
        min_stamp_s: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return ``(tip_position_world, R_world_from_tip)`` or ``None``.

        This is the shape the SC controller's ``sc_tip_pose_from_tcp`` returns,
        so it is the natural drop-in at the call sites -- with the crucial
        difference that this one can answer "I do not know" and does so
        whenever the observation is stale, under-determined, or inconsistent
        with the known plug geometry.

        Runtime callers must pass both ``now_s`` and ``max_age_s`` in the image
        stamps' clock domain (normally ROS simulation time); omitting them is
        intended only for offline evaluation.
        """

        estimate = self.estimate_multiview(
            views,
            now_s=now_s,
            max_age_s=max_age_s,
            min_stamp_s=min_stamp_s,
        )
        if estimate is None:
            return None
        return estimate.position_world, estimate.rotation_world_from_plug


def load_sc_plug_pose_estimator(
    weights_path: str | Path | None = None,
    **kwargs,
) -> ScPlugPoseEstimator | None:
    """Build an :class:`ScPlugPoseEstimator`, or ``None`` if no checkpoint.

    Returning ``None`` rather than raising lets a controller log "SC plug-pose
    model unavailable" once and refuse the insertion, which is the intended
    fail-closed behaviour when a build ships without the checkpoint.  Passing
    an explicit ``weights_path`` that does not exist still raises, because that
    is a configuration error rather than an absent optional model.
    """

    if weights_path is None:
        weights_path = default_sc_plug_pose_weights()
        if weights_path is None:
            return None
    return ScPlugPoseEstimator(weights_path, **kwargs)


__all__ = [
    "SC_MAX_KEYPOINT_RMSE_M",
    "SC_MAX_REPROJECTION_ERROR_PX",
    "SC_PLUG_POSE_WEIGHTS_BASENAME",
    "SC_PLUG_POSE_WEIGHTS_ENV",
    "PlugPoseEstimate",
    "PlugPoseView",
    "ScPlugPoseEstimator",
    "default_sc_plug_pose_weights",
    "load_sc_plug_pose_estimator",
]

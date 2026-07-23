from argparse import Namespace

import mujoco

from RL.scene_env import SceneEnvConfig, _compile_scene_model
from RL.student_teacher.gate0_contact_jam import _jam_window


def _args():
    return Namespace(
        jam_window_s=1.2,
        jam_max_progress_mm=1.0,
        jam_min_force_n=8.0,
        jam_depth_min_mm=5.0,
        jam_depth_max_mm=9.0,
    )


def _trace(force=9.0, contacts=1):
    return [
        {
            "step": i + 1,
            "time_s": i * 0.1,
            "depth_m": 0.0065,
            "insertion_depth_mm": 6.5,
            "mouth_gap_mm": 0.0,
            "lateral_mm": 1.0,
            "axis_deg": 1.0,
            "force_n": force,
            "plug_port_contacts": contacts,
        }
        for i in range(14)
    ]


def test_sustained_contact_stall_classifier_requires_joint_evidence():
    jam = _jam_window(_trace(), "timeout", _args())
    assert jam is not None
    assert jam["kind"] == "sustained_contact_stall"
    assert _jam_window(_trace(force=0.0), "timeout", _args()) is None
    assert _jam_window(_trace(contacts=0), "timeout", _args()) is None


def test_compiled_ridge_keeps_centered_clearance_after_scaling():
    cfg = SceneEnvConfig(compiled_variant_seed=20260712)
    model, diag = _compile_scene_model(cfg)
    ridge = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "aic_random_contact_ridge_pos")
    plug = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "contact_collision_1")
    ridge_inner = model.geom_pos[ridge, 0] - model.geom_size[ridge, 0]
    geometric_clearance = ridge_inner - model.geom_size[plug, 0]
    assert abs(geometric_clearance - diag["contact_ridge_clearance_m"]) < 1e-12
    assert geometric_clearance > max(cfg.random_contact_pair_margin_range_m)
    top = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "aic_random_contact_ridge_top")
    plug_other = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "non_contact_collision")
    ridge_inner_y = model.geom_pos[top, 1] - model.geom_size[top, 1]
    plug_union_y = max(
        abs(model.geom_pos[i, 1]) + model.geom_size[i, 1]
        for i in (plug, plug_other)
    )
    geometric_clearance_y = ridge_inner_y - plug_union_y
    assert abs(geometric_clearance_y - diag["contact_ridge_clearance_m"]) < 1e-12
    assert geometric_clearance_y > max(cfg.random_contact_pair_margin_range_m)

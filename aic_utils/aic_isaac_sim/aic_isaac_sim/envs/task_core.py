"""Vectorized geometry, reward, and termination shared by the Isaac task.

This module only depends on PyTorch. It mirrors the geometry-first MuJoCo
contract in ``RL.scene_env`` and the active reward in ``RL.reward``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TaskThresholds:
    seated_depth_m: float = 0.0458
    success_depth_norm: float = 0.99
    success_axial_tol_m: float = 0.003
    success_lateral_tol_m: float = 0.005
    success_axis_tol_rad: float = 0.035
    success_roll_tol_rad: float = 0.15
    success_max_overinsert_m: float = 0.001
    bad_collision_depth_gate: float = 0.45
    bad_collision_axis_rad: float = 0.35
    bad_collision_roll_rad: float = 0.35
    bad_collision_overinsert_m: float = 0.002
    force_abort_n: float = 60.0
    force_abort_hard_n: float = 120.0
    force_abort_dwell_steps: int = 3


@dataclass(frozen=True)
class RewardWeights:
    w_depth: float = 30.0
    xy_ref: float = 0.006
    w_xy: float = 0.35
    axis_free_rad: float = 0.05
    axis_ref_rad: float = 0.15
    w_axis: float = 0.15
    force_free_n: float = 12.0
    force_ref_n: float = 20.0
    w_force: float = 0.20
    lateral_free_n: float = 3.0
    lateral_ref_n: float = 10.0
    w_lateral: float = 0.20
    w_action: float = 0.02
    success_bonus: float = 50.0
    bad_collision_penalty: float = 25.0
    force_abort_penalty: float = 25.0
    timeout_penalty: float = 15.0


def _unit(value: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(eps)


def quat_rotate_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by unit quaternions in wxyz convention."""
    xyz = q[..., 1:]
    uv = torch.cross(xyz, v, dim=-1)
    uuv = torch.cross(xyz, uv, dim=-1)
    return v + 2.0 * (q[..., :1] * uv + uuv)


def insertion_geometry(
    tip_pos_w: torch.Tensor,
    tail_pos_w: torch.Tensor,
    tip_quat_w: torch.Tensor,
    port_pos_w: torch.Tensor,
    port_quat_w: torch.Tensor,
    seated_depth_m: float,
) -> dict[str, torch.Tensor]:
    basis_x = torch.zeros_like(port_pos_w)
    basis_x[:, 0] = 1.0
    basis_z = torch.zeros_like(port_pos_w)
    basis_z[:, 2] = 1.0
    insert_axis = _unit(quat_rotate_wxyz(port_quat_w, basis_z))
    lat_x_raw = quat_rotate_wxyz(port_quat_w, basis_x)
    lat_x = _unit(lat_x_raw - (lat_x_raw * insert_axis).sum(-1, keepdim=True) * insert_axis)
    lat_y = _unit(torch.cross(insert_axis, lat_x, dim=-1))

    seated_tip = port_pos_w + seated_depth_m * insert_axis
    delta_seated = tip_pos_w - seated_tip
    retract_dir = -insert_axis
    axial_error = (delta_seated * retract_dir).sum(-1)
    lateral_xy = torch.stack(
        ((delta_seated * lat_x).sum(-1), (delta_seated * lat_y).sum(-1)), dim=-1
    )
    lateral_error = torch.linalg.vector_norm(lateral_xy, dim=-1)
    depth_m = ((tip_pos_w - port_pos_w) * insert_axis).sum(-1)
    depth_norm = (depth_m / seated_depth_m).clamp(0.0, 1.0)
    overinsert_m = (depth_m - seated_depth_m).clamp_min(0.0)

    plug_axis = _unit(tip_pos_w - tail_pos_w)
    axis_error = torch.acos((plug_axis * insert_axis).sum(-1).clamp(-1.0, 1.0))
    tip_local_x = quat_rotate_wxyz(tip_quat_w, basis_x)
    plug_roll = _unit(tip_local_x - (tip_local_x * insert_axis).sum(-1, keepdim=True) * insert_axis)
    roll_error = torch.acos((plug_roll * lat_x).sum(-1).clamp(-1.0, 1.0))

    return {
        "insert_axis": insert_axis,
        "lat_x": lat_x,
        "lat_y": lat_y,
        "seated_tip": seated_tip,
        "lateral_xy": lateral_xy,
        "lateral_error": lateral_error,
        "axial_error": axial_error,
        "depth_m": depth_m,
        "depth_norm": depth_norm,
        "overinsert_m": overinsert_m,
        "axis_error": axis_error,
        "roll_error": roll_error,
    }


def termination_masks(
    geometry: dict[str, torch.Tensor],
    force_w: torch.Tensor,
    force_over_count: torch.Tensor,
    thresholds: TaskThresholds,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    insert_axis = geometry["insert_axis"]
    f_axial = (force_w * insert_axis).sum(-1)
    f_lateral = torch.linalg.vector_norm(force_w - f_axial.unsqueeze(-1) * insert_axis, dim=-1)
    f_norm = torch.linalg.vector_norm(force_w, dim=-1)
    f_peak = torch.maximum(torch.maximum(f_axial.abs(), f_lateral), f_norm)
    force_over_count = torch.where(
        f_peak > thresholds.force_abort_n,
        force_over_count + 1,
        torch.zeros_like(force_over_count),
    )

    success = (
        (geometry["depth_norm"] >= thresholds.success_depth_norm)
        & (geometry["axial_error"].abs() < thresholds.success_axial_tol_m)
        & (geometry["lateral_error"] < thresholds.success_lateral_tol_m)
        & (geometry["axis_error"] <= thresholds.success_axis_tol_rad)
        & (geometry["roll_error"] <= thresholds.success_roll_tol_rad)
        & (geometry["overinsert_m"] <= thresholds.success_max_overinsert_m)
    )
    bad_collision = (
        (geometry["overinsert_m"] > thresholds.bad_collision_overinsert_m)
        | (
            (geometry["depth_norm"] >= thresholds.bad_collision_depth_gate)
            & (
                (geometry["axis_error"] > thresholds.bad_collision_axis_rad)
                | (geometry["roll_error"] > thresholds.bad_collision_roll_rad)
            )
        )
    )
    force_abort = (f_peak > thresholds.force_abort_hard_n) | (
        force_over_count >= thresholds.force_abort_dwell_steps
    )
    return success, bad_collision, force_abort, force_over_count


def compute_reward(
    geometry: dict[str, torch.Tensor],
    force_w: torch.Tensor,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    previous_depth_norm: torch.Tensor,
    success: torch.Tensor,
    bad_collision: torch.Tensor,
    force_abort: torch.Tensor,
    timeout: torch.Tensor,
    weights: RewardWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    depth = weights.w_depth * (geometry["depth_norm"] - previous_depth_norm)
    xy = -weights.w_xy * (geometry["lateral_error"] / weights.xy_ref).clamp(0.0, 6.0)
    alignment = torch.maximum(geometry["axis_error"], geometry["roll_error"])
    axis = -weights.w_axis * (
        (alignment - weights.axis_free_rad).clamp_min(0.0) / weights.axis_ref_rad
    ).clamp(0.0, 2.0)

    f_axial = (force_w * geometry["insert_axis"]).sum(-1)
    f_lateral = torch.linalg.vector_norm(
        force_w - f_axial.unsqueeze(-1) * geometry["insert_axis"], dim=-1
    )
    force = -weights.w_force * (
        (f_axial.abs() - weights.force_free_n).clamp_min(0.0) / weights.force_ref_n
    ).clamp(0.0, 3.0)
    lateral = -weights.w_lateral * (
        (f_lateral - weights.lateral_free_n).clamp_min(0.0) / weights.lateral_ref_n
    ).clamp(0.0, 3.0)
    action = -weights.w_action * torch.linalg.vector_norm(actions - previous_actions, dim=-1).clamp(0.0, 4.0)
    done = (
        success.float() * weights.success_bonus
        - bad_collision.float() * weights.bad_collision_penalty
        - force_abort.float() * weights.force_abort_penalty
        - timeout.float() * weights.timeout_penalty
    )
    total = depth + xy + axis + force + lateral + action + done
    return total, {
        "depth": depth,
        "xy": xy,
        "axis": axis,
        "force": force,
        "lateral": lateral,
        "action": action,
        "done": done,
    }


__all__ = [
    "RewardWeights",
    "TaskThresholds",
    "compute_reward",
    "insertion_geometry",
    "quat_rotate_wxyz",
    "termination_masks",
]

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from engineai_rl_lab.tasks.mimic.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)

def undesired_contacts(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float
) -> torch.Tensor:
    """惩罚非期望身体部位发生的接触。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 获取各body当前接触力的净力大小 shape: [N, num_bodies]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # 取最近一步的接触力幅值 shape: [N, num_bodies]
    force_magnitude = torch.norm(net_contact_forces[:, 0, sensor_cfg.body_ids, :], dim=-1)
    # 超过阈值则视为非期望接触 shape: [N, num_bodies]
    is_contact = force_magnitude > threshold
    # 对所有非期望接触 body 求和，返回惩罚值 shape: [N]
    return torch.sum(is_contact, dim=-1).float()


def feet_collision_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_distance_x: float = 0.10,
    min_distance_y: float = 0.05,
    std: float = 0.05,
) -> torch.Tensor:
    """
    惩罚左右脚掌距离过近的情况：
    - X轴距离小于 min_distance_x (10cm) 时惩罚
    - Y轴距离小于 min_distance_y (5cm) 时惩罚
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_names = command.cfg.body_names
    left_idx  = body_names.index("LINK_ANKLE_ROLL_L")
    right_idx = body_names.index("LINK_ANKLE_ROLL_R")
    # robot_body_pos_w: [num_envs, num_bodies, 3]
    left_foot_pos  = command.robot_body_pos_w[:, left_idx,  :]  # [N, 3]
    right_foot_pos = command.robot_body_pos_w[:, right_idx, :]  # [N, 3]
    # 分别计算 X、Y 轴绝对距离  [N]
    dist_x = torch.abs(left_foot_pos[:, 0] - right_foot_pos[:, 0])
    dist_y = torch.abs(left_foot_pos[:, 1] - right_foot_pos[:, 1])
    # 各轴低于阈值的不足量  [N]
    error_x = torch.clamp(min_distance_x - dist_x, min=0.0)
    error_y = torch.clamp(min_distance_y - dist_y, min=0.0)
    # 两轴误差求和  [N]
    total_error = error_x + error_y
    # 惩罚值（weight 设为负数）
    penalty = torch.exp(total_error / std) - 1.0
    return penalty
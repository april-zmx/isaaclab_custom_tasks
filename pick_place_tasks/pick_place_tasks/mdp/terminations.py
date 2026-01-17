# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms,subtract_frame_transforms
from isaaclab.assets import Articulation, RigidObject
from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.04,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object to reach the goal position. Defaults to 0.02.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    """
    robot: Articulation = env.scene[robot_cfg.name]
    object_data: RigidObjectData = env.scene[object_cfg.name].data
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[..., :3]
    # des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)

    object_position_w = object_data.root_pos_w
    object_position,_ = subtract_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_position_w)
    object_position[:,0] = object_position[:,0]+0.25
    object_position[:,2] = object_position[:,2]-0.10
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_b - object_position, dim=1)
    # print(distance)
    # print("11:",object_position_w)
    # print("22:",robot.data.root_state_w[:, :3])
    # print("33:",robot.data.root_state_w[:, 3:7])
    # print("44:",object_position)
    # print("55:",des_pos_b)
    return distance < threshold

def object_in_box(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    object_data: RigidObject = env.scene[asset_cfg.name]
    object_position_w = object_data.data.root_pos_w-env.scene.env_origins


    # 计算各轴向的布尔条件
    x_cond = (object_position_w[:, 0] > 0.55) & (object_position_w[:, 0] < 0.85)
    y_cond = (object_position_w[:, 1] > -0.1) & (object_position_w[:, 1] < 0.1)
    z_cond = (object_position_w[:, 2] > 0.6) & (object_position_w[:, 2] < 0.65)
    
    # 直接返回布尔张量
    return x_cond & y_cond & z_cond


def task_done_pick_place(
    env: ManagerBasedRLEnv,
    task_link_name: str = "",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    plate_object_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    left_wrist_max_x: float = 0.26,
    min_x: float = 0.40,
    max_x: float = 0.85,
    min_y: float = 0.35,
    max_y: float = 0.60,
    max_height: float = 1.10,
    min_vel: float = 0.20,
) -> torch.Tensor:
    """Determine if the object placement task is complete.

    This function checks whether all success conditions for the task have been met:
    1. object is within the target x/y range
    2. object is below a minimum height
    3. object velocity is below threshold
    4. Right robot wrist is retracted back towards body (past a given x pos threshold)

    Args:
        env: The RL environment instance.
        object_cfg: Configuration for the object entity.
        left_wrist_max_x: Maximum x position of the left wrist for task completion.
        min_x: Minimum x position of the object for task completion.
        max_x: Maximum x position of the object for task completion.
        min_y: Minimum y position of the object for task completion.
        max_y: Maximum y position of the object for task completion.
        max_height: Maximum height (z position) of the object for task completion.
        min_vel: Minimum velocity magnitude of the object for task completion.

    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    if task_link_name == "":
        raise ValueError("task_link_name must be provided to task_done_pick_place")

    # Get object entity from the scene
    object: RigidObject = env.scene[object_cfg.name]
    plate_object: RigidObject = env.scene[plate_object_cfg.name]

    # Extract object position relative to environment origin
    object_x = object.data.root_pos_w[:, 0]# - env.scene.env_origins[:, 0]
    object_y = object.data.root_pos_w[:, 1]# - env.scene.env_origins[:, 1]
    object_height = object.data.root_pos_w[:, 2]# - env.scene.env_origins[:, 2]
    object_vel = torch.abs(object.data.root_vel_w)

    # Get object position relative to plate
    object_x = object_x - plate_object.data.root_pos_w[:, 0]
    object_y = object_y - plate_object.data.root_pos_w[:, 1]
    object_height = object_height - plate_object.data.root_pos_w[:, 2]

    # Get right wrist position relative to environment origin
    robot_body_pos_w = env.scene["robot"].data.body_pos_w
    left_eef_idx = env.scene["robot"].data.body_names.index(task_link_name)
    left_wrist_x = robot_body_pos_w[:, left_eef_idx, 0] - env.scene.env_origins[:, 0]

    # Check all success conditions and combine with logical AND
    done = object_x < max_x
    done = torch.logical_and(done, object_x > min_x)
    done = torch.logical_and(done, object_y < max_y)
    done = torch.logical_and(done, object_y > min_y)
    done = torch.logical_and(done, object_height < max_height)
    # done = torch.logical_and(done, left_wrist_x < left_wrist_max_x)
    done = torch.logical_and(done, object_vel[:, 0] < min_vel)
    done = torch.logical_and(done, object_vel[:, 1] < min_vel)
    done = torch.logical_and(done, object_vel[:, 2] < min_vel)

    return done
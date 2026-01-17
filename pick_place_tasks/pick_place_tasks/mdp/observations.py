# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.assets import Articulation, RigidObject
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_pose_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_b, object_quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, object.data.root_pos_w, object.data.root_quat_w
    )
    object_pos_b = torch.clip(object_pos_b, min=-2., max=2.)
    return torch.cat((object_pos_b, object_quat_b), dim=1)

def object_position_in_ee_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    tcp_position_w = ee_frame.data.target_pos_w[..., 0, :].clone()
    tcp_quat_w = ee_frame.data.target_quat_w[..., 0, :].clone()
    object_pos_b, _ = math_utils.subtract_frame_transforms(
        tcp_position_w, tcp_quat_w, object.data.root_pos_w
    )
    object_pos_b = torch.clip(object_pos_b, min=-2., max=2.)
    return object_pos_b

def ee_frame_pos(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]
    tcp_rest_position_w = ee_frame.data.target_pos_w[..., 0, :].clone()
    tcp_rest_position, _ = math_utils.subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], tcp_rest_position_w)
    return tcp_rest_position

def ee_frame_quat(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]
    tcp_rest_position_w = ee_frame.data.target_pos_w[..., 0, :].clone()
    tcp_rest_orientation_w = ee_frame.data.target_quat_w[..., 0, :].clone()
    _, tcp_rest_orientation = math_utils.subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], tcp_rest_position_w, tcp_rest_orientation_w)
    return tcp_rest_orientation

def single_joint_pos_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    selected_indices = env.action_manager.get_term("arm_action")._joint_ids
    return asset.data.joint_pos[:, selected_indices] - asset.data.default_joint_pos[:, selected_indices]

def single_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    selected_indices = env.action_manager.get_term("arm_action")._joint_ids
    return asset.data.joint_pos[:, selected_indices]

def single_joint_pos_limit_normalized(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The joint positions of the asset normalized with the asset's joint limits.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their normalized positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    selected_indices = env.action_manager.get_term("arm_action")._joint_ids
    return math_utils.scale_transform(
        asset.data.joint_pos[:, selected_indices],
        asset.data.soft_joint_pos_limits[:, selected_indices, 0],
        asset.data.soft_joint_pos_limits[:, selected_indices, 1],
    )

def single_joint_vel_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    selected_indices = env.action_manager.get_term("arm_action")._joint_ids
    return asset.data.joint_vel[:, selected_indices] - asset.data.default_joint_vel[:, selected_indices]

def single_joint_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    selected_indices = env.action_manager.get_term("arm_action")._joint_ids
    return asset.data.joint_vel[:, selected_indices]


def gripper_pos(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Obtain the versatile gripper position of both Gripper and Suction Cup.
    """
    robot: Articulation = env.scene[robot_cfg.name]

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        # Handle multiple surface grippers by concatenating their states
        gripper_states = []
        for gripper_name, surface_gripper in env.scene.surface_grippers.items():
            gripper_states.append(surface_gripper.state.view(-1, 1))

        if len(gripper_states) == 1:
            return gripper_states[0]
        else:
            return torch.cat(gripper_states, dim=1)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            assert len(gripper_joint_ids) == 2, "Observation gripper_pos only support parallel gripper for now"
            finger_joint_1 = robot.data.joint_pos[:, gripper_joint_ids[0]].clone().unsqueeze(1)
            finger_joint_2 = -1 * robot.data.joint_pos[:, gripper_joint_ids[1]].clone().unsqueeze(1)
            return finger_joint_1 - finger_joint_2
        else:
            raise NotImplementedError("[Error] Cannot find gripper_joint_names in the environment config")


# def image(
#     env: ManagerBasedRLEnv,
#     sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
#     data_type: str = "rgb",
# ) -> torch.Tensor:
#     """Images of a specific datatype from the camera sensor.

#     If the flag :attr:`normalize` is True, post-processing of the images are performed based on their
#     data-types:

#     - "rgb": Scales the image to (0, 1) and subtracts with the mean of the current image batch.
#     - "depth" or "distance_to_camera" or "distance_to_plane": Replaces infinity values with zero.

#     Args:
#         env: The environment the cameras are placed within.
#         sensor_cfg: The desired sensor to read from. Defaults to SceneEntityCfg("tiled_camera").
#         data_type: The data type to pull from the desired camera. Defaults to "rgb".
#         convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
#             This is used only when the data type is "distance_to_camera". Defaults to False.
#         normalize: Whether to normalize the images. This depends on the selected data type.
#             Defaults to True.

#     Returns:
#         The images produced at the last time-step
#     """
#     # extract the used quantities (to enable type-hinting)
#     sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]

#     # obtain the input image
#     images = sensor.data.output[data_type]

#     return images.clone()


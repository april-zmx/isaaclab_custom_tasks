# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import os
from dataclasses import MISSING
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from . import mdp
from .grasp_env_cfg import GraspEnvCfg, PickPlaceSceneCfg


CUSTOM_USD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


@configclass
class VisionPickPlaceSceneCfg(PickPlaceSceneCfg):
    
    # cameras
    left_wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_camera/left_cam",
        update_period=0.02,
        height=168,
        width=224,
        data_types=["rgb"],
        colorize_semantic_segmentation=True,
        colorize_instance_id_segmentation=True,
        colorize_instance_segmentation=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=5.7, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.05, 0.0, 0.1), rot=(-0.55721, -0.43534, 0.43534, 0.55721), convention="opengl"
        ),
    )
    right_wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_camera/right_cam",
        update_period=0.02,
        height=168,
        width=224,
        data_types=["rgb"],
        colorize_semantic_segmentation=True,
        colorize_instance_id_segmentation=True,
        colorize_instance_segmentation=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=5.7, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.05, 0.0, 0.1), rot=(-0.55721, -0.43534, 0.43534, 0.55721),convention="opengl"
        ),
    )
    head_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/camera_link2/head_cam",
        update_period=0.02,
        height=168,
        width=224,
        data_types=["rgb","depth","semantic_segmentation"],
        semantic_filter="class:object",
        colorize_semantic_segmentation=True,
        colorize_instance_id_segmentation=True,
        colorize_instance_segmentation=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=5.7, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.4, 0.0, 0.0), rot=(0.69638, 0.12279, -0.12278, -0.69635), convention="opengl"
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        joint_pos = ObsTerm(func=mdp.single_joint_pos)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)
        

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CamerasCfg(ObsGroup):
        """Observations for policy group with state values."""

        head_cam = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("head_cam"), "data_type": "rgb", "normalize": False}
        )
        left_wrist_cam = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("left_wrist_cam"), "data_type": "rgb", "normalize": False}
        )
        right_wrist_cam = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("right_wrist_cam"), "data_type": "rgb", "normalize": False}
        )
        # table_cam_segmentation = ObsTerm(
        #     func=mdp.image,
        #     params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "semantic_segmentation", "normalize": True},
        # )
        # table_cam_normals = ObsTerm(
        #     func=mdp.image,
        #     params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "normals", "normalize": True},
        # )
        # table_cam_depth = ObsTerm(
        #     func=mdp.image,
        #     params={
        #         "sensor_cfg": SceneEntityCfg("table_cam"),
        #         "data_type": "distance_to_image_plane",
        #         "normalize": True,
        #     },
        # )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    cameras: CamerasCfg = CamerasCfg()


@configclass
class OpenpiGraspEnvCfg(GraspEnvCfg):
    """Configuration for the lifting environment."""
    # Scene settings
    scene: VisionPickPlaceSceneCfg = VisionPickPlaceSceneCfg(num_envs=4096, env_spacing=2.5)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()

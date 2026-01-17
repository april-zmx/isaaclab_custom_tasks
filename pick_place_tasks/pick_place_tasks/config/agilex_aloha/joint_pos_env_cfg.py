# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import os
from isaaclab.utils import configclass
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.sensors import ContactSensorCfg

from ...grasp_env_cfg import GraspEnvCfg
from ...grasp_vision_env_cfg import VisionGraspEnvCfg
from ...grasp_openpi_env_cfg import OpenpiGraspEnvCfg
from ... import mdp
from .agilex_aloha import AGILEX_ALOHA_CFG


CUSTOM_USD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@configclass
class AgilexGraspEnvCfg(GraspEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = AGILEX_ALOHA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/(fr_link.*|fl_link.*)",
            update_period=0.0,
            history_length=0,
            debug_vis=False,
            # filter_prim_paths_expr=[
            #     "{ENV_REGEX_NS}/Table",
            #     "{ENV_REGEX_NS}/Plate",
            #     "{ENV_REGEX_NS}/High_object",
            #     "{ENV_REGEX_NS}/Low_Object",
            # ],
        )

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", 
            joint_names=[
                'fl_joint1', 
                'fl_joint2', 
                'fl_joint3', 
                'fl_joint4', 
                'fl_joint5', 
                'fl_joint6', 
                ],
            use_default_offset=False,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["fl_joint7", "fl_joint8"],
            open_command_expr={"fl_joint7": 0.04, "fl_joint8": 0.04},
            close_command_expr={"fl_joint7": 0.0, "fl_joint8": 0.0},
            debug_vis=False,
        )

        # utilities for gripper status check
        self.gripper_joint_names = ["fl_joint7", "fl_joint8"]
        self.gripper_open_val = 0.04
        self.gripper_threshold = 0.005

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/fl_base_link",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/fl_ee_link",
                    name="left_ee_frame",
                    # offset=OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.707, 0.0, 0.0, -0.707)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/fr_ee_link",
                    name="right_ee_frame",
                    # offset=OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.707, 0.0, 0.0, -0.707)),
                ),
            ],
        )


@configclass
class AgilexVisionGraspEnvCfg(VisionGraspEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = AGILEX_ALOHA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/(fr_link.*|fl_link.*)",
            update_period=0.0,
            history_length=0,
            debug_vis=False,
            # filter_prim_paths_expr=[
            #     "{ENV_REGEX_NS}/Table",
            #     "{ENV_REGEX_NS}/Plate",
            #     "{ENV_REGEX_NS}/High_object",
            #     "{ENV_REGEX_NS}/Low_Object",
            # ],
        )

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", 
            joint_names=[
                'fl_joint1', 
                'fl_joint2', 
                'fl_joint3', 
                'fl_joint4', 
                'fl_joint5', 
                'fl_joint6', 
                ],
            use_default_offset=False,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["fl_joint7", "fl_joint8"],
            open_command_expr={"fl_joint7": 0.04, "fl_joint8": 0.04},
            close_command_expr={"fl_joint7": 0.0, "fl_joint8": 0.0},
            debug_vis=False,
        )

        # utilities for gripper status check
        self.gripper_joint_names = ["fl_joint7", "fl_joint8"]
        self.gripper_open_val = 0.04
        self.gripper_threshold = 0.005

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/fl_base_link",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/fl_ee_link",
                    name="left_ee_frame",
                    # offset=OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.707, 0.0, 0.0, -0.707)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/fr_ee_link",
                    name="right_ee_frame",
                    # offset=OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.707, 0.0, 0.0, -0.707)),
                ),
            ],
        )


@configclass
class AgilexOpenpiGraspEnvCfg(OpenpiGraspEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = AGILEX_ALOHA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/(fr_link.*|fl_link.*)",
            update_period=0.0,
            history_length=0,
            debug_vis=False,
            # filter_prim_paths_expr=[
            #     "{ENV_REGEX_NS}/Table",
            #     "{ENV_REGEX_NS}/Plate",
            #     "{ENV_REGEX_NS}/High_object",
            #     "{ENV_REGEX_NS}/Low_Object",
            # ],
        )

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", 
            joint_names=[
                'fl_joint1', 
                'fl_joint2', 
                'fl_joint3', 
                'fl_joint4', 
                'fl_joint5', 
                'fl_joint6', 
                ],
            use_default_offset=False,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["fl_joint7", "fl_joint8"],
            open_command_expr={"fl_joint7": 0.04, "fl_joint8": 0.04},
            close_command_expr={"fl_joint7": 0.0, "fl_joint8": 0.0},
            debug_vis=False,
        )

        # utilities for gripper status check
        self.gripper_joint_names = ["fl_joint7", "fl_joint8"]
        self.gripper_open_val = 0.04
        self.gripper_threshold = 0.005

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/fl_base_link",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/fl_ee_link",
                    name="left_ee_frame",
                    # offset=OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.707, 0.0, 0.0, -0.707)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/fr_ee_link",
                    name="right_ee_frame",
                    # offset=OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.707, 0.0, 0.0, -0.707)),
                ),
            ],
        )

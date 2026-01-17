# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from isaaclab.utils import configclass

from ... import mdp
from . import joint_pos_env_cfg


@configclass
class AgilexGraspEnvCfg(joint_pos_env_cfg.AgilexGraspEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot", 
            joint_names=[
                'fl_joint1', 
                'fl_joint2', 
                'fl_joint3', 
                'fl_joint4', 
                'fl_joint5', 
                'fl_joint6', 
                ],
            use_zero_offset=True,
            scale=0.25,
            clip={".*": (-1., 1.)},
            preserve_order=True,
        )


@configclass
class AgilexVisionGraspEnvCfg(joint_pos_env_cfg.VisionGraspEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot", 
            joint_names=[
                'fl_joint1', 
                'fl_joint2', 
                'fl_joint3', 
                'fl_joint4', 
                'fl_joint5', 
                'fl_joint6', 
                ],
            use_zero_offset=True,
            scale=0.25,
            clip={".*": (-1., 1.)},
            preserve_order=True,
        )


@configclass
class AgilexOpenpiGraspEnvCfg(joint_pos_env_cfg.OpenpiGraspEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot", 
            joint_names=[
                'fl_joint1', 
                'fl_joint2', 
                'fl_joint3', 
                'fl_joint4', 
                'fl_joint5', 
                'fl_joint6', 
                ],
            use_zero_offset=True,
            scale=0.25,
            clip={".*": (-1., 1.)},
            preserve_order=True,
        )

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
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.sim.spawners.sensors.sensors_cfg import PinholeCameraCfg 
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from . import mdp


CUSTOM_USD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


@configclass
class PickPlaceSceneCfg(InteractiveSceneCfg):
    """Configuration for the pick place scene with a robot and many objects.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """
    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING

    contact_sensor: ContactSensorCfg = MISSING

    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg(),
                          init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)))

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Table
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.5, 0.8, 0.74),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled = True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
        ),
        init_state = RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.2, 0.37), rot=(1, 0, 0, 0)),
    ) 

    # plate
    plate = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Plate",
        init_state= RigidObjectCfg.InitialStateCfg(
            pos=(-0.5, 0.0, 0.78),
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
        spawn = sim_utils.UsdFileCfg(   
            usd_path=os.path.join(CUSTOM_USD_DIR, "objects/plate/plate.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4, 
                solver_velocity_iteration_count=0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
        ), 
    )
    
    # objects
    high_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/High_object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            random_choice=True,
            assets_cfg=[
                sim_utils.CylinderCfg(
                    radius=0.03,
                    height=0.2,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False,),
                    physics_material = sim_utils.RigidBodyMaterialCfg(
                        static_friction = 1.0, 
                        dynamic_friction = 1.0,
                        restitution = 0.0,
                    ),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),                        
                ),
            ],
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.05, -0.05, 0.84)),
    )

    medium_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Medium_Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            random_choice=True,
            assets_cfg=[
                sim_utils.CylinderCfg(
                    radius=0.03,
                    height=0.1,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False,),
                    physics_material = sim_utils.RigidBodyMaterialCfg(
                        static_friction = 1.0, 
                        dynamic_friction = 1.0,
                        restitution = 0.0,
                    ),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),                        
                ),
            ],
            mass_props=sim_utils.MassPropertiesCfg(mass=0.25),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.05, 0.08, 0.79)),
    )

    low_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Low_Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            random_choice=True,
            assets_cfg=[
                UsdFileCfg(
                    usd_path=os.path.join(CUSTOM_USD_DIR, "objects/banana/banana.usd"),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                        max_angular_velocity=1000.0,
                        max_linear_velocity=1000.0,
                        max_depenetration_velocity=5.0,
                        disable_gravity=False,
                    ),
                ),
            ],
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.79)),
    )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    arm_action: mdp.JointPositionActionCfg | mdp.RelativeJointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        joint_pos = ObsTerm(func=mdp.single_joint_pos_rel)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)
        joint_vel = ObsTerm(func=mdp.single_joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        object_pose = ObsTerm(func=mdp.object_pose_in_robot_root_frame, params={
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("medium_object"),
        })
        object_pos = ObsTerm(func=mdp.object_position_in_ee_frame, params={
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg": SceneEntityCfg("medium_object"),
        })

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_robot = EventTerm(
        func=mdp.reset_joints_by_offset, 
        mode="reset", 
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        }
    )

    reset_high_object_pose_with_medium = EventTerm(
        func=mdp.reset_root_state_uniform_multi_bind_object,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.3, 0.1), 
                "y": (0.0, 0.2), 
                "yaw": (-3.14, 3.14)
            },
            "velocity_range": {},
            "parent_asset_cfg": SceneEntityCfg("high_object"),
            "child_asset_cfg": SceneEntityCfg("medium_object"),
        },
    )

    reset_low_object_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.4, 0.1), 
                "y": (-0.05, 0.4), 
                "yaw": (-3.14, 3.14)
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("low_object"),
        },
    )

    reset_plate_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05), 
                "y": (-0.1, 0.2), 
                "z": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("plate"),
        },
    )



@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # success = RewTerm(
    #     func=mdp.is_terminated_term,
    #     weight=10.0,
    #     params={
    #         "term_keys": "success"
    #     }
    # )

    # failed = RewTerm(
    #     func=mdp.is_terminated_term,
    #     weight=-10.0,
    #     params={
    #         "term_keys": "object_dropping"
    #     }
    # )

    # collision_penalty = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-0.1,
    #     params={
    #         "threshold": 1.0,
    #         "sensor_cfg": SceneEntityCfg(
    #             name="contact_sensor", 
    #             body_names="(fr_link.*|fl_link.*)",
    #         )
    #     }
    # )

    time_penalty = RewTerm(func=mdp.time_consumption, weight=-0.01)

    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        weight=15.0,
        params={
            "minimal_height": 0.74 + 0.1,  # 0.74 is the height of table
            "object_cfg": SceneEntityCfg("medium_object"),
        }
    )

    # only left arm
    object_ee_distance = RewTerm(
        func=mdp.object_ee_distance,
        weight=1.0,
        params={
            "std": 0.25,
            "object_cfg": SceneEntityCfg("medium_object"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        }
    )

    # only left arm
    goal_distance = RewTerm(
        func=mdp.object_goal_distance,
        weight=10.0,
        params={
            "std": 0.25,
            "minimal_height": 0.74 + 0.1,  # 0.74 is the height of table
            "place_object_cfg": SceneEntityCfg("plate"),
            "object_cfg": SceneEntityCfg("medium_object"),
        }
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("medium_object")}
    )

    # success = DoneTerm(
    #     func=mdp.task_done_pick_place, 
    #     params={
    #         "task_link_name": "fl_ee_link",
    #         "object_cfg": SceneEntityCfg("medium_object"),
    #         "plate_object_cfg": SceneEntityCfg("plate"),
    #         "left_wrist_max_x": 0.2,
    #         "min_x": -0.05,
    #         "max_x":  0.05,
    #         "min_y": -0.05,
    #         "max_y":  0.05,
    #         "max_height": 0.11,
    #         "min_vel": 0.10,
    #     }
    # )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass

##
# Environment configuration
##


@configclass
class GraspEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""
    # Scene settings
    scene: PickPlaceSceneCfg = PickPlaceSceneCfg(num_envs=4096, env_spacing=2.5)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    
    # MDP settingsReplicator:Annotators
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 3
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1./90.  # 100Hz
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        # self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.gpu_max_rigid_patch_count = 2**20
        self.sim.physx.friction_correlation_distance = 0.0064

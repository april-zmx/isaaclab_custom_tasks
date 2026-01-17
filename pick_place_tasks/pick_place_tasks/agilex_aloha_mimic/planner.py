# ********************** CuroboPlanner (optional) **********************
from curobo.types.math import Pose as CuroboPose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)
from curobo.util import logger

import time
import torch
import yaml
import numpy as np
# import transforms3d as t3d
from typing import List, Dict, Union

import isaaclab.utils.math as math_utils
# 源于robotwin2.0

class CuroboPlanner:

    def __init__(
        self,
        active_joints_name: List[str],
        # all_joints: List[str],
        world_config: Dict,
        yml_path: Union[str, None] = None,
    ):
        super().__init__()
        logger.setup_logger(level="error", logger_name="'curobo")

        if yml_path != None:
            self.yml_path = yml_path
        else:
            raise ValueError("[Planner.py]: CuroboPlanner yml_path is None!")
        self.active_joints_name = active_joints_name
        # self.all_joints = all_joints

        # # translate from baselink to arm's base
        # with open(self.yml_path, "r") as f:
        #     yml_data = yaml.safe_load(f)
        # self.frame_bias = yml_data["planner"]["frame_bias"]

        # motion generation
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.yml_path,
            world_config,
            interpolation_dt=0.02,
            num_trajopt_seeds=32,
            # rotation_threshold=0.2,
        )

        self.motion_gen = MotionGen(motion_gen_config)
        self.motion_gen.warmup()
        # motion_gen_config = MotionGenConfig.load_from_robot_config(
        #     self.yml_path,
        #     world_config,
        #     interpolation_dt=0.02,
        #     num_trajopt_seeds=1,
        #     num_graph_seeds=1,
        # )
        # self.motion_gen_batch = MotionGen(motion_gen_config)
        # self.motion_gen_batch.warmup(batch=10)

    def plan_path(
        self,
        curr_joint_pos: np.ndarray,
        target_gripper_pose: np.ndarray,
        constraint_pose=None,
        arms_tag=None,
    ):

        goal_pose_of_gripper = CuroboPose.from_list(target_gripper_pose.tolist())
        start_joint_states = JointState.from_position(
            torch.tensor(curr_joint_pos).cuda().reshape(1, -1),
            joint_names=self.active_joints_name,
        )
        # plan
        plan_config = MotionGenPlanConfig(max_attempts=10)
        if constraint_pose is not None:
            pose_cost_metric = PoseCostMetric(
                hold_partial_pose=True,
                hold_vec_weight=self.motion_gen.tensor_args.to_device(constraint_pose),
            )
            plan_config.pose_cost_metric = pose_cost_metric

        self.motion_gen.reset(reset_seed=True)  # 运行的代码
        result = self.motion_gen.plan_single(start_joint_states, goal_pose_of_gripper, plan_config)
        # result = self.motion_gen.plan_goalset(start_joint_states, goal_pose_of_gripper, plan_config)
        # traj = result.get_interpolated_plan()

        # output
        res_result = dict()
        if result.success.item() == False:
            res_result["status"] = "Fail"
            print(result.status)
            return res_result
        else:
            res_result["status"] = "Success"
            res_result["position"] = np.array(result.interpolated_plan.position.to("cpu"))
            res_result["velocity"] = np.array(result.interpolated_plan.velocity.to("cpu"))
            return res_result
        
    def plan_ee_pose(self, start_joint_pos, target_pose_b, constraint_pose=None, candidate_target_poses=None):
        """
        规划从当前关节到目标世界位姿的路径
        target_pose_w: [x, y, z, qw, qx, qy, qz]
        """
        # 1. 将世界位姿转换为机器人基座坐标系（如果基座不是原点）
        # 这里简化处理，假设 target_pose_w 已经是相对机器人基座的
        
        target_pose = CuroboPose(
            position=target_pose_b[:3].clone(),
            quaternion=target_pose_b[3:].clone() # Curobo 通常使用 wxyz
        )
        candidate_target_poses = [
            CuroboPose(
                position=pose_b[:3].clone(),
                quaternion=pose_b[3:].clone()
            ) for pose_b in candidate_target_poses
        ] if candidate_target_poses is not None else []
        target_poses = [target_pose] + candidate_target_poses
        
        # 2. 构造起始状态
        start_state = JointState.from_position(
            torch.tensor(start_joint_pos, device="cuda").view(1, -1),
            joint_names=self.active_joints_name
        )
        
        # 3. 规划
        plan_config = MotionGenPlanConfig(max_attempts=60)
        if constraint_pose is not None:
            pose_cost_metric = PoseCostMetric(
                hold_partial_pose=True,
                hold_vec_weight=self.motion_gen.tensor_args.to_device(constraint_pose),
            )
            plan_config.pose_cost_metric = pose_cost_metric
        # self.motion_gen.reset(reset_seed=True)

        for i, target_pose in enumerate(target_poses):
            result = self.motion_gen.plan_single(start_state, target_pose, plan_config)
            if result.success:
                break  # 找到成功的规划就退出循环
        
        res_result = {}
        if result.success.item():
            # 提取插值后的路径
            res_result["position"] = result.interpolated_plan.position.cpu().numpy()
            res_result["status"] = "Success"
        else:
            res_result["status"] = "Failure"
        return res_result.get("position", None), i

    # def plan_batch(
    #     self,
    #     curr_joint_pos,
    #     target_gripper_pose_list,
    #     constraint_pose=None,
    #     arms_tag=None,
    # ):
    #     """
    #     Plan a batch of trajectories for multiple target poses.

    #     Input:
    #         - curr_joint_pos: List of current joint angles (1 x n)
    #         - target_gripper_pose_list: List of target poses [sapien.Pose, sapien.Pose, ...]

    #     Output:
    #         - result['status']: numpy array of string values indicating "Success"/"Fail" for each pose
    #         - result['position']: numpy array of joint positions with shape (n x m x l)
    #             where n is number of target poses, m is number of waypoints, l is number of joints
    #         - result['velocity']: numpy array of joint velocities with same shape as position
    #     """

    #     num_poses = len(target_gripper_pose_list)
    #     # transformation from world to arm's base
    #     world_base_pose = np.concatenate([
    #         np.array(self.robot_origion_pose.p),
    #         np.array(self.robot_origion_pose.q),
    #     ])
    #     poses_list = []
    #     for target_gripper_pose in target_gripper_pose_list:
    #         world_target_pose = np.concatenate([np.array(target_gripper_pose.p), np.array(target_gripper_pose.q)])
    #         base_target_pose_p, base_target_pose_q = self._trans_from_world_to_base(
    #             world_base_pose, world_target_pose)
    #         base_target_pose_list = list(base_target_pose_p) + list(base_target_pose_q)
    #         base_target_pose_list[0] += self.frame_bias[0]
    #         base_target_pose_list[1] += self.frame_bias[1]
    #         base_target_pose_list[2] += self.frame_bias[2]
    #         poses_list.append(base_target_pose_list)

    #     poses_cuda = torch.tensor(poses_list, dtype=torch.float32).cuda()
    #     #
    #     goal_pose_of_gripper = CuroboPose(poses_cuda[:, :3], poses_cuda[:, 3:])
    #     joint_indices = [self.all_joints.index(name) for name in self.active_joints_name if name in self.all_joints]
    #     joint_angles = [curr_joint_pos[index] for index in joint_indices]
    #     joint_angles = [round(angle, 5) for angle in joint_angles]  # avoid the precision problem
    #     joint_angles_cuda = (torch.tensor(joint_angles, dtype=torch.float32).cuda().reshape(1, -1))
    #     joint_angles_cuda = torch.cat([joint_angles_cuda] * num_poses, dim=0)
    #     start_joint_states = JointState.from_position(joint_angles_cuda, joint_names=self.active_joints_name)
    #     # plan
    #     c_start_time = time.time()
    #     plan_config = MotionGenPlanConfig(max_attempts=10)
    #     if constraint_pose is not None:
    #         pose_cost_metric = PoseCostMetric(
    #             hold_partial_pose=True,
    #             hold_vec_weight=self.motion_gen.tensor_args.to_device(constraint_pose),
    #         )
    #         plan_config.pose_cost_metric = pose_cost_metric

    #     self.motion_gen.reset(reset_seed=True)
    #     try:
    #         result = self.motion_gen_batch.plan_batch(start_joint_states, goal_pose_of_gripper, plan_config)
    #     except Exception as e:
    #         return {"status": ["Failure" for i in range(10)]}

    #     # output
    #     res_result = dict()
    #     # Convert boolean success values to "Success"/"Failure" strings
    #     success_array = result.success.cpu().numpy()
    #     status_array = np.array(["Success" if s else "Failure" for s in success_array], dtype=object)
    #     res_result["status"] = status_array

    #     if np.all(res_result["status"] == "Failure"):
    #         return res_result

    #     res_result["position"] = np.array(result.interpolated_plan.position.to("cpu"))
    #     res_result["velocity"] = np.array(result.interpolated_plan.velocity.to("cpu"))
    #     return res_result

    # def plan_grippers(self, now_val, target_val):
    #     num_step = 50
    #     dis_val = target_val - now_val
    #     step = dis_val / num_step
    #     res = {}
    #     vals = np.linspace(now_val, target_val, num_step)
    #     vals = vals.astype(np.float32)
    #     res["num_step"] = num_step
    #     res["per_step"] = step
    #     res["result"] = vals
    #     return res

    # def _trans_from_world_to_base(self, base_pose, target_pose):
    #     '''
    #         transform target pose from world frame to base frame
    #         base_pose: np.array([x, y, z, qw, qx, qy, qz])
    #         target_pose: np.array([x, y, z, qw, qx, qy, qz])
    #     '''
    #     base_p, base_q = base_pose[0:3], base_pose[3:]
    #     target_p, target_q = target_pose[0:3], target_pose[3:]
    #     rel_p = target_p - base_p
    #     wRb = t3d.quaternions.quat2mat(base_q)
    #     wRt = t3d.quaternions.quat2mat(target_q)
    #     result_p = wRb.T @ rel_p
    #     result_q = t3d.quaternions.mat2quat(wRb.T @ wRt)
    #     return result_p, result_q
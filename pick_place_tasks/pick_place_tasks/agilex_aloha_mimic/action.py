import os
from typing import Optional, List
# 给定目标位置，生成包含逆解的控制序列
from planner import CuroboPlanner
import copy
# import transforms3d as t3d
import yaml
import numpy as np
import torch
from PIL import Image
from recorder import RobotDataRecorder
import time

from isaaclab.scene import InteractiveScene
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply_inverse, quat_apply, subtract_frame_transforms

CONFIG_PATH = os.path.dirname(os.path.abspath(__file__))

class Pose:
    def __init__(self,p=[0, -0.65, 0],q=[0.707, 0, 0, 0.707]):
        # 初始化p和q属性
        self.p = p
        self.q = q

class Action:
    def __init__(self, scene: InteractiveScene, sim):

        self.recorder = RobotDataRecorder(output_dir="robot_recordings11.27")

        #导入配置文件 
        # config_path = os.path.join(CONFIG_PATH, "curobo_config/config.yml")
        # with open(config_path, "r") as f:
        #     robot_yml_data = yaml.safe_load(f)
        # self.left_gripper_bias = robot_yml_data['gripper_bias']
        # self.right_gripper_bias = robot_yml_data['gripper_bias']
        # self.left_delta_matrix = np.array(robot_yml_data['delta_matrix'])
        # self.left_inv_delta_matrix = np.linalg.inv(self.left_delta_matrix)
        # self.right_delta_matrix = np.array(robot_yml_data['delta_matrix'])
        # self.right_inv_delta_matrix = np.linalg.inv(self.left_delta_matrix)

        self.robot: Articulation = scene["robot"]
        self.env_index = 0

        # 生成碰撞世界
        world_config = {
            "cuboid": {
                "table": {
                    "dims": [0.7, 2, 0.04],  # x, y, z
                    "pose": [
                        0.0,
                        0.0,
                        0.74,
                        1,
                        0,
                        0,
                        0.0,
                    ],  # x, y, z, qw, qx, qy, qz
                },
            }
        }

        # 初始化左臂规划器
        left_arm_config = os.path.join(CONFIG_PATH, "curobo_config/curobo_left.yml")
        with open(left_arm_config, "r") as f:
            config = yaml.safe_load(f)
        left_arm_base_link = config["robot_cfg"]["kinematics"]["base_link"]
        self.left_arm_joints_names = [
            "fl_joint1","fl_joint2","fl_joint3","fl_joint4","fl_joint5","fl_joint6"
        ]
        arm_base_index, _ = self.robot.find_bodies([left_arm_base_link])
        self.init_left_arm_base_pose = self.robot.data.body_pose_w[self.env_index, arm_base_index].clone()
        self.left_joint_indices, _ = self.robot.find_joints(self.left_arm_joints_names)
        self.left_planner = CuroboPlanner(
            self.left_arm_joints_names,
            world_config,
            yml_path=left_arm_config,
        )

        # 初始化右臂规划器
        right_arm_config = os.path.join(CONFIG_PATH, "curobo_config/curobo_right.yml")
        with open(right_arm_config, "r") as f:
            config = yaml.safe_load(f)
        right_arm_base_link = config["robot_cfg"]["kinematics"]["base_link"]
        self.right_arm_joints_names = [
            "fr_joint1","fr_joint2","fr_joint3","fr_joint4","fr_joint5","fr_joint6"
        ]
        arm_base_index, _ = self.robot.find_bodies([right_arm_base_link])
        self.init_right_arm_base_pose = self.robot.data.body_pose_w[self.env_index, arm_base_index].clone()
        self.right_joint_indices, _ = self.robot.find_joints(self.right_arm_joints_names)
        self.right_planner = CuroboPlanner(
            self.right_arm_joints_names,
            world_config,
            yml_path=right_arm_config
        )

        self.left_grapper_state = 'close'
        self.right_grapper_state = 'open'
        
        self.scene = scene
        self.sim = sim

        self.recorder_left_arm_joints_name = [
            'fl_joint1','fl_joint2','fl_joint3','fl_joint4','fl_joint5','fl_joint6','fl_joint7'
        ]
        self.recorder_right_arm_joints_name = [
            'fr_joint1','fr_joint2','fr_joint3','fr_joint4','fr_joint5','fr_joint6','fr_joint7'
        ]

    def record(self):
        # 记录当前机械臂状态
        self._record_camera_images()
        self._record_joint_data()


    def plan(
            self, 
            target_pose: torch.Tensor, 
            constraint_pose: Optional[List[float]] = None, 
            candidate_target_poses: Optional[torch.Tensor] = None
        ) -> List[Optional[torch.Tensor]]:
        # 坐标变换到相对于arm_base
        pose = subtract_frame_transforms(
            t01=self.init_left_arm_base_pose[self.env_index, :3].view(1, -1),
            q01=self.init_left_arm_base_pose[self.env_index, 3:].view(1, -1),
            t02=target_pose[self.env_index, :3].view(1, -1),
            q02=target_pose[self.env_index, 3:].view(1, -1),
        )
        target_pose_b = torch.cat(pose, dim=1)[self.env_index]
        if candidate_target_poses is not None:
            pose = subtract_frame_transforms(
                t01=self.init_left_arm_base_pose[self.env_index, :3].view(-1, 3).repeat(candidate_target_poses.shape[1], 1),
                q01=self.init_left_arm_base_pose[self.env_index, 3:].view(-1, 4).repeat(candidate_target_poses.shape[1], 1),
                t02=candidate_target_poses[self.env_index, :, :3].view(-1, 3),
                q02=candidate_target_poses[self.env_index, :, 3:].view(-1, 4),
            )
            candidate_target_poses_b = torch.cat(pose, dim=1).view(-1, 7)
        else:
            candidate_target_poses_b = None

        joint_pos = self.robot.data.joint_pos.clone().cpu().numpy()[self.env_index]

        # constraint_pose = [1, 1, 1, 1, 1, 0]
        trajs, pose_idx = self.left_planner.plan_ee_pose(
            start_joint_pos=joint_pos[self.left_joint_indices],
            target_pose_b=target_pose_b,
            constraint_pose=constraint_pose,
            candidate_target_poses=candidate_target_poses_b if candidate_target_poses_b is not None else None,
        )

        if trajs is not None:
            success_target_pose = target_pose if pose_idx == 0 or candidate_target_poses is None else candidate_target_poses[:, pose_idx - 1]
        else:
            success_target_pose = None
        return [trajs, success_target_pose]
    
    def move_to_pos(self,des_pose,arm_tag):
        # 给定curobo规划结果，控制指定的机械臂移动
        sim_dt = self.sim.get_physics_dt()
        pos_cmd = torch.from_numpy(des_pose['position']).to(device='cuda')
        vel_cmd = torch.from_numpy(des_pose['velocity']).to(device='cuda')
        
        for index in range(pos_cmd.shape[0]):
            if(arm_tag=='left_arm'):
                self.scene["robot"].set_joint_position_target(pos_cmd[index,:],joint_ids=[6,14,18,22,26,30])#[1,11,18,22,26,30
                self.scene["robot"].set_joint_velocity_target(vel_cmd[index,:],joint_ids=[6,14,18,22,26,30])
            else:
                self.scene["robot"].set_joint_position_target(pos_cmd[index,:],joint_ids=[7,15,19,23,27,31])#[1,11,18,22,26,30
                self.scene["robot"].set_joint_velocity_target(vel_cmd[index,:],joint_ids=[7,15,19,23,27,31])
            self._record_camera_images()
            self._record_joint_data()
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(sim_dt)
    
    def together_move_to_pose(self,pose_left,pose_right):
        # 控制双臂共同移动到指定的位置
        sim_dt = self.sim.get_physics_dt()
        joint_pos = copy.deepcopy(self.scene['robot'].data.joint_pos.cpu().squeeze().numpy())
        des_pose_left = self.left_planner.plan_path(target_gripper_pose=pose_left,curr_joint_pos=joint_pos)
        des_pose_right = self.right_planner.plan_path(target_gripper_pose=pose_right,curr_joint_pos=joint_pos)
        left_success = False if des_pose_left['status']=='Fail' else True
        right_success = False if des_pose_right['status']=='Fail' else True
        if left_success:
            pos_cmd_left = torch.from_numpy(des_pose_left['position']).to(device='cuda')
            vel_cmd_left = torch.from_numpy(des_pose_left['velocity']).to(device='cuda')
        if right_success:
            pos_cmd_right = torch.from_numpy(des_pose_right['position']).to(device='cuda')
            vel_cmd_right = torch.from_numpy(des_pose_right['velocity']).to(device='cuda')
        now_left_id = 0
        now_right_id = 0
        left_n_step = pos_cmd_left.shape[0] if left_success else 0
        right_n_step = pos_cmd_right.shape[0] if right_success else 0
        while now_left_id < left_n_step or now_right_id < right_n_step:
            if (left_success and now_left_id < left_n_step
                    and (not right_success or now_left_id / left_n_step <= now_right_id / right_n_step)):
                self.scene["robot"].set_joint_position_target(pos_cmd_left[now_left_id,:],joint_ids=[1,11,18,22,26,30])#[1,11,18,22,26,30
                self.scene["robot"].set_joint_velocity_target(vel_cmd_left[now_left_id,:],joint_ids=[1,11,18,22,26,30])
                now_left_id += 1  
            if (right_success and now_right_id < right_n_step
                    and (not left_success or now_right_id / right_n_step <= now_left_id / left_n_step)):
                self.scene["robot"].set_joint_position_target(pos_cmd_right[now_right_id,:],joint_ids=[3,13,19,23,27,31])#[1,11,18,22,26,30
                self.scene["robot"].set_joint_velocity_target(vel_cmd_right[now_right_id,:],joint_ids=[3,13,19,23,27,31])
                now_right_id += 1 
            self._record_camera_images()
            self._record_joint_data()
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(sim_dt)  

    def together_open_grapper(self):
        # 控制夹爪同时打开
        sim_dt = self.sim.get_physics_dt()
        des_grapper = self.left_planner.plan_grippers(now_val=0.0,target_val=15.0)
        vel_cmd = torch.from_numpy(des_grapper['result']).to(device='cuda')
        step = vel_cmd.shape[0]
        now_step = 0
        while now_step<step:
            self.scene["robot"].set_joint_velocity_target(vel_cmd[now_step],joint_ids=[34])
            self.scene["robot"].set_joint_velocity_target(vel_cmd[now_step],joint_ids=[35])
            self.scene["robot"].set_joint_velocity_target(vel_cmd[now_step],joint_ids=[36])
            self.scene["robot"].set_joint_velocity_target(vel_cmd[now_step],joint_ids=[37])
            now_step+=1
        self.right_grapper_state = 'open'
        self.leftgrapper_state = 'open'
        self._record_camera_images()
        self._record_joint_data()
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(sim_dt)
    
    def together_close_grapper(self):
        # 控制夹爪同时闭合
        sim_dt = self.sim.get_physics_dt()
        des_grapper = self.left_planner.plan_grippers(now_val=15.0,target_val=0.0)
        vel_cmd = torch.from_numpy(des_grapper['result']).to(device='cuda')
        step = vel_cmd.shape[0]
        now_step = 0
        while now_step<step:
            self.scene["robot"].set_joint_velocity_target(vel_cmd[now_step],joint_ids=[34])
            self.scene["robot"].set_joint_velocity_target(vel_cmd[now_step],joint_ids=[35])
            self.scene["robot"].set_joint_velocity_target(vel_cmd[now_step],joint_ids=[36])
            self.scene["robot"].set_joint_velocity_target(vel_cmd[now_step],joint_ids=[37])
            now_step+=1
        self.right_grapper_state = 'close'
        self.leftgrapper_state = 'close'
        self._record_camera_images()
        self._record_joint_data()
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(sim_dt)

    def move_to_homestate(self,arm_tag,pos):
        # 移动到初始位置
        joint_pos = copy.deepcopy(self.scene["robot"].data.joint_pos.cpu().squeeze().numpy())
        if(arm_tag=='left_arm'):
            des_pos_pre = self.left_planner.plan_path(target_gripper_pose=pos,curr_joint_pos=joint_pos)
            des_grapper = self.left_planner.plan_grippers(now_val=15.0,target_val=0.0)
            self.move_to_pos(des_pos_pre,arm_tag)
            if(self.left_grapper_state=='open'):
                self.close_grapper(arm_tag=arm_tag,des_grapper=des_grapper)
            if des_pos_pre['status'] == 'Fail':
                print('plan failed!!')
                return
        else:
            des_pos_pre = self.right_planner.plan_path(target_gripper_pose=pos,curr_joint_pos=joint_pos)
            des_grapper = self.right_planner.plan_grippers(now_val=15.0,target_val=0.0)
            self.move_to_pos(des_pos_pre,arm_tag)
            if(self.right_grapper_state=='open'):
                self.close_grapper(arm_tag=arm_tag,des_grapper=des_grapper)
            if des_pos_pre['status'] == 'Fail':
                print('plan failed!!')
                return
        
    def open_grapper(self,arm_tag,des_grapper):
        # 打开夹爪
        sim_dt = self.sim.get_physics_dt()
        vel_cmd = torch.from_numpy(des_grapper['result']).to(device='cuda')
        for index in range(des_grapper['num_step']):
            if(arm_tag=='left_arm'):
                self.scene["robot"].set_joint_velocity_target(vel_cmd[index],joint_ids=[34])
                self.scene["robot"].set_joint_velocity_target(vel_cmd[index],joint_ids=[35])
                self.left_grapper_state = 'open'
            else:
                self.scene["robot"].set_joint_velocity_target(vel_cmd[index],joint_ids=[36])
                self.scene["robot"].set_joint_velocity_target(vel_cmd[index],joint_ids=[37])
                self.right_grapper_state = 'open'
            self._record_camera_images()
            self._record_joint_data()
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(sim_dt)

    def close_grapper(self, arm_tag, des_grapper):
        # 关闭夹爪
        sim_dt = self.sim.get_physics_dt()
        vel_cmd = torch.from_numpy(des_grapper['result']).to(device='cuda')
        for index in range(des_grapper['num_step']):
            if(arm_tag=='left_arm'):
                self.scene["robot"].set_joint_velocity_target(vel_cmd[index],joint_ids=[34])
                self.scene["robot"].set_joint_velocity_target(vel_cmd[index],joint_ids=[35])
                self.left_grapper_state = 'close'
            else:
                self.scene["robot"].set_joint_velocity_target(vel_cmd[index],joint_ids=[36])
                self.scene["robot"].set_joint_velocity_target(vel_cmd[index],joint_ids=[37])
                self.right_grapper_state = 'close'
            self._record_camera_images()
            self._record_joint_data()
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(sim_dt)

    def take_grasp_action(self, target_gripper_pose: torch.Tensor, arm_tag: str):
        # 打开夹爪
        if(arm_tag=='left_arm'):
            des_grapper = self.left_planner.plan_grippers(now_val=0.0,target_val=15.0)
        else:
            des_grapper = self.right_planner.plan_grippers(now_val=0.0,target_val=15.0)
        self.open_grapper(arm_tag,des_grapper)

        # 坐标变换到相对于arm_base
        if arm_tag == "left_arm":
            pose = subtract_frame_transforms(
                t01=self.init_left_arm_base_pose[self.env_index, :3].view(1, -1),
                q01=self.init_left_arm_base_pose[self.env_index, 3:].view(1, -1),
                t02=target_gripper_pose[self.env_index, :3].view(1, -1),
                q02=target_gripper_pose[self.env_index, 3:].view(1, -1),
            )
            grasp_pose = torch.cat(pose, dim=1).squeeze(0)
        else:
            pose = subtract_frame_transforms(
                t01=self.init_right_arm_base_pose[self.env_index, :3].view(1, -1),
                q01=self.init_right_arm_base_pose[self.env_index, 3:].view(1, -1),
                t02=target_gripper_pose[self.env_index, :3].view(1, -1),
                q02=target_gripper_pose[self.env_index, 3:].view(1, -1),
            )
            grasp_pose = torch.cat(pose, dim=1).squeeze(0)

        joint_pos = self.robot.data.joint_pos.clone().cpu().numpy()[self.env_index]

        # 移动到目标位置
        if(arm_tag=='left_arm'):
            des_pos_pre = self.left_planner.plan_path(
                target_gripper_pose=grasp_pose.cpu().numpy(), 
                curr_joint_pos=joint_pos[self.left_joint_indices]
            )
            if des_pos_pre['status'] == 'Fail':
                print('grasp left plan failed!!')
                return
        else:
            des_pos_pre = self.right_planner.plan_path(
                target_gripper_pose=grasp_pose.cpu().numpy(), 
                curr_joint_pos=joint_pos[self.right_joint_indices]
            )
            if des_pos_pre['status'] == 'Fail':
                print('grasp right plan failed!!')
                return
        self.move_to_pos(des_pos_pre, arm_tag)

        # 关闭夹爪 
        if arm_tag=='left_arm':
            des_grapper = self.left_planner.plan_grippers(now_val=15.0,target_val=0.0)
        else:
            des_grapper = self.right_planner.plan_grippers(now_val=15.0,target_val=0.0)
        self.close_grapper(arm_tag,des_grapper)

        # 向上抬起
        joint_pos = self.robot.data.joint_pos.clone().cpu().numpy()[self.env_index]
        lift_pose = grasp_pose.clone()
        lift_pose[2] += 0.2

        constraint_pose = [1, 1, 1, 1, 1, 0]
        if arm_tag=='left_arm':
            des_lift_pos = self.left_planner.plan_path(
                target_gripper_pose=lift_pose.cpu().numpy(),
                curr_joint_pos=joint_pos[self.left_joint_indices],
                constraint_pose=constraint_pose
            )
            if des_lift_pos['status'] == 'Fail':
                print('lift left plan failed!!')
                return
        else:
            des_lift_pos = self.right_planner.plan_path(
                target_gripper_pose=lift_pose.cpu().numpy(), 
                curr_joint_pos=joint_pos[self.right_joint_indices],
                constraint_pose=constraint_pose
            )
            if des_lift_pos['status'] == 'Fail':
                print('lift right plan failed!!')
                return
        self.move_to_pos(des_lift_pos, arm_tag)
        print('plan finished!! reset!!')
    
    def take_place_action(self, target_place_pose: torch.Tensor, arm_tag: str):
        # 坐标变换到相对于arm_base
        if arm_tag == "left_arm":
            pose = subtract_frame_transforms(
                t01=self.init_left_arm_base_pose[self.env_index, :3].view(1, -1),
                q01=self.init_left_arm_base_pose[self.env_index, 3:].view(1, -1),
                t02=target_place_pose[self.env_index, :3].view(1, -1),
                q02=target_place_pose[self.env_index, 3:].view(1, -1),
            )
            place_pose = torch.cat(pose, dim=1)[self.env_index]
        else:
            pose = subtract_frame_transforms(
                t01=self.init_right_arm_base_pose[self.env_index, :3].view(1, -1),
                q01=self.init_right_arm_base_pose[self.env_index, 3:].view(1, -1),
                t02=target_place_pose[self.env_index, :3].view(1, -1),
                q02=target_place_pose[self.env_index, 3:].view(1, -1),
            )
            place_pose = torch.cat(pose, dim=1).squeeze(0)

        joint_pos = self.robot.data.joint_pos.clone().cpu().numpy()[self.env_index]

        # 执行放置动作
        # 移动到目标位置
        if arm_tag == 'left_arm':
            des_pos = self.left_planner.plan_path(
                target_gripper_pose=place_pose.cpu().numpy(),
                curr_joint_pos=joint_pos[self.left_joint_indices]
            ) 
            if des_pos['status'] == 'Fail':
                print('take_place_action plan failed!! reset!!')
                return              
        else:
            des_pos = self.right_planner.plan_path(
                target_gripper_pose=place_pose.cpu().numpy(),
                curr_joint_pos=joint_pos[self.right_joint_indices]
            )   
            if des_pos['status'] == 'Fail':
                print('take_place_action plan failed!! reset!!')
                return
        self.move_to_pos(des_pos, arm_tag)
        
        # 打开夹爪
        if arm_tag == 'left_arm':
            des_grapper = self.left_planner.plan_grippers(now_val=0.0,target_val=15.0)
        else:
            des_grapper = self.right_planner.plan_grippers(now_val=0.0,target_val=15.0)
        self.open_grapper(arm_tag,des_grapper)
        if arm_tag == 'left_arm':
            des_grapper = self.left_planner.plan_grippers(now_val=15.0,target_val=0.0)
        else:
            des_grapper = self.right_planner.plan_grippers(now_val=15.0,target_val=0.0)
        self.close_grapper(arm_tag,des_grapper)
    
    def move_by_displacement(self,pos,arm_tag):
        # 控制机械臂移动到指定的位置
        joint_pos = copy.deepcopy(self.scene["robot"].data.joint_pos.cpu().squeeze().numpy())
        if(arm_tag=='left_arm'):
            des_pos_pre = self.left_planner.plan_path(target_gripper_pose=pos,curr_joint_pos=joint_pos)
            if des_pos_pre['status'] == 'Fail':
                print('plan failed!!')
                return
            self.move_to_pos(des_pos_pre,arm_tag)
        else:
            des_pos_pre = self.right_planner.plan_path(target_gripper_pose=pos,curr_joint_pos=joint_pos)
            if des_pos_pre['status'] == 'Fail':
                print('plan failed!!')
                return  
            self.move_to_pos(des_pos_pre,arm_tag)  

    def start_recording(self):
        self.recorder.start_recording()    

    def stop_recording(self):
        self.recorder.stop_recording()    

    def save_recording(self):
        self.recorder.save_data()
    
    def clear_recording(self):
        self.recorder.clear_data()

    def _record_camera_images(self):
        """
        记录所有相机的图像数据
        """
        # 记录左相机图像
        left_camera = self.scene["left_wrist_cam"].data.output["rgb"][0]
        left_rgb_np = left_camera.cpu().numpy().astype(np.uint8)  # 转换为uint8格式
        self.recorder.record_image('left_camera', left_rgb_np)
        
        # 记录右相机图像
        right_camera = self.scene["right_wrist_cam"].data.output["rgb"][0]
        right_rgb_np = right_camera.cpu().numpy().astype(np.uint8)  # 转换为uint8格式
        self.recorder.record_image('right_camera', right_rgb_np)
        
        # 记录头部相机图像
        head_camera = self.scene["head_cam"].data.output["rgb"][0]
        head_rgb_np = head_camera.cpu().numpy().astype(np.uint8)  # 转换为uint8格式
        self.recorder.record_image('head_camera', head_rgb_np)


        head_camera_mask = self.scene["head_cam"].data.output["semantic_segmentation"][0]
        seg_np = head_camera_mask.cpu().numpy()
        head_mask_np = seg_np[..., 0].astype(np.bool_)  # 转换为uint8格
        self.recorder.record_image('head_camera_mask', head_mask_np)

    # 3. 添加记录关节数据的函数
    def _record_joint_data(self):
        """
        记录机械臂关节数据，包括过滤后的特定关节数据和全部关节数据
        """
        # 获取所有关节名称和索引映射
        all_joint_names = self.scene['robot'].joint_names
        joint_name_to_index = {name: i for i, name in enumerate(all_joint_names)}

        # 获取所有关节位置数据
        all_joint_pos = self.scene['robot'].data.joint_pos.cpu().squeeze().numpy()
        joint_pos_dict_left = {}
        joint_pos_dict_right = {}
        
        # 只记录指定的左手臂和右手臂关节
        for joint_name in self.recorder_left_arm_joints_name:
            if joint_name in joint_name_to_index:
                idx = joint_name_to_index[joint_name]
                joint_pos_dict_left[joint_name] = all_joint_pos[idx]

        # 只记录指定的右手臂关节
        for joint_name in self.recorder_right_arm_joints_name:
            if joint_name in joint_name_to_index:
                idx = joint_name_to_index[joint_name]
                joint_pos_dict_right[joint_name] = all_joint_pos[idx]

        # 记录过滤后的关节数据
        merged_joint_dict = {**joint_pos_dict_left, **joint_pos_dict_right}
        self.recorder.record_joint_state(merged_joint_dict)
        self.recorder.record_action(merged_joint_dict)  # 注意：这里使用位置数据作为动作，可能需要修改
        
        self.recorder.timestamps.append(time.time() - self.recorder.record_start_time)


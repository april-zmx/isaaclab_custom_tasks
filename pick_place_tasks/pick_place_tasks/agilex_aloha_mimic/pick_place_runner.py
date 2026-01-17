import argparse
from isaaclab.app import AppLauncher

# 1. 配置启动参数
parser = argparse.ArgumentParser(description="Isaac-Agilex Pick-Place Task via env.step (CUDA Optimized)")
parser.add_argument("--num_envs", type=int, default=1, help="环境数量")
parser.add_argument("--is_visual", action="store_true", default=False, help="是否可视化")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动仿真
app_launcher = AppLauncher(num_envs=1, enable_cameras=True)
simulation_app = app_launcher.app

import os
import torch
import numpy as np
import gymnasium as gym
from typing import Tuple, Dict, Any, List

import isaaclab.utils.math as math_utils
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers
from action import Action

import sys
sys.path.append("./source")
import pick_place_tasks
from pick_place_tasks.agilex_aloha_mimic.action import Action

class PickPlaceRunner:
    """
    主运行器：管理环境、状态机并调用 env.step 执行动作。
    所有物理计算和张量操作均在 CUDA 上完成。
    """
    def __init__(self, env_name="Isaac-Agilex-PickPlace-ThreeObjects-v2"):
        self.device = "cuda"
        # 解析环境配置
        env_cfg = parse_env_cfg(env_name, device=self.device, num_envs=args_cli.num_envs)
        self.env = gym.make(env_name, cfg=env_cfg)
        
        # 获取场景引用
        self.unwrapped_env: ManagerBasedRLEnv = self.env.unwrapped
        self.scene = self.unwrapped_env.scene
        self.robot = self.scene['robot']

        # Initialize a visualization marker for target gripper poses
        try:
            marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/World/Visuals/target_pose")
            marker_cfg.markers["frame"].scale = (0.06, 0.06, 0.06)
            self.target_pose_visualizer = VisualizationMarkers(marker_cfg)
        except Exception:
            self.target_pose_visualizer = None

        # 初始化 Action 类（内部集成 Recorder 和 Planner）
        self.action_controller = Action(self.scene, simulation_app)
        self.recorder = self.action_controller.recorder

        self.is_success = torch.zeros((args_cli.num_envs,), device=self.device, dtype=torch.bool)

    def reset(self):
        """重置环境并初始化状态"""
        self.env.reset()
        self.is_success.zero_()
        for _ in range(10):
            self.env.step(torch.from_numpy(self.env.action_space.sample()* 0.).to(self.device))

    def get_object_pose_w(self, object_name: str) -> torch.Tensor:
        """获取物体在世界坐标系下的位姿 [x, y, z, qw, qx, qy, qz]，返回 CUDA Tensor"""
        obj_sensor = self.scene[object_name]
        # 直接使用 IsaacLab 的 Tensor 数据 (M, 3) 和 (M, 4)
        obj_pose_w = obj_sensor.data.root_pose_w.clone()

        # 旋转物体坐标系对齐末端夹爪
        offset_quat = math_utils.quat_from_angle_axis(
            torch.tensor([np.pi / 2.], device=self.device, dtype=torch.float32),
            torch.tensor([[0., 1., 0.]], device=self.device, dtype=torch.float32),
        )
        obj_pose_w = math_utils.combine_frame_transforms(
            obj_pose_w[:, :3], obj_pose_w[:, 3:], q12=offset_quat)
        return torch.cat(obj_pose_w, dim=1)

    def execute_trajectory(self, joint_trajectory: torch.Tensor, gripper_val: float):
        """
        通过 env.step 循环执行轨迹。
        joint_trajectory 应为 (N, 6) 的 CUDA Tensor。
        """
        num_steps = joint_trajectory.shape[0]
        # 预先准备好夹爪张量
        gripper_tensor = torch.full((1, 1), gripper_val, device=self.device)
        
        for i in range(num_steps):
            # 在 GPU 上拼接动作: [1, 6] + [1, 1] -> [1, 7]
            waypoint = joint_trajectory[i:i+1] 
            action_vec = torch.cat([waypoint, gripper_tensor], dim=-1)
            
            # 执行环境步进
            obs, reward, terminated, truncated, info = self.env.step(action_vec)
            # breakpoint()

            self.is_success.copy_(terminated)
            
            # 同步记录数据 (内部会处理 CPU 转换用于保存)
            self._record_step(action_vec, obs, reward, info)
            
            if terminated or truncated:
                break

    def _record_step(self, action, obs, reward, info):
        """记录逻辑，仅在必要时转换为 numpy"""
        # 记录关节状态和动作
        joint_names = self.unwrapped_env.action_manager.get_term("arm_action")._joint_names
        joint_names = joint_names + ["fl_joint7"]

        joint_pos = obs["policy"].cpu().numpy()[0]
        action = action.cpu().numpy()[0]
        joint_state_dict, action_dict = {}, {}
        for k, v1, v2 in zip(joint_names, joint_pos, action):
            joint_state_dict[k] = v1
            action_dict[k] = v2

        # 记录过滤后的关节数据
        self.recorder.record_joint_state(joint_state_dict)
        self.recorder.record_action(action_dict)
        
        self.recorder.timestamps.append((self.unwrapped_env.episode_length_buf * self.unwrapped_env.step_dt).cpu().numpy()[0])

        # 记录左相机图像
        left_camera = obs["cameras"]["left_wrist_cam"][0]
        left_rgb_np = left_camera.cpu().numpy().astype(np.uint8)  # 转换为uint8格式
        self.recorder.record_image('left_camera', left_rgb_np)
        
        # 记录右相机图像
        right_camera = obs["cameras"]["right_wrist_cam"][0]
        right_rgb_np = right_camera.cpu().numpy().astype(np.uint8)  # 转换为uint8格式
        self.recorder.record_image('right_camera', right_rgb_np)
        
        # 记录头部相机图像
        head_camera = obs["cameras"]["head_cam"][0]
        head_rgb_np = head_camera.cpu().numpy().astype(np.uint8)  # 转换为uint8格式
        self.recorder.record_image('head_camera', head_rgb_np)

    def run_pick_place(self, object_key: str, target_place_key: str):
        """
        执行完整的 Pick 和 Place 流程。
        target_place_pose: [x, y, z, qw, qx, qy, qz] CUDA Tensor
        """
        print(f"--- 正在执行任务: 抓取 {object_key} ---")
        self.recorder.start_recording()
        
        # 1. 获取物体当前位置 (CUDA Tensor)
        obj_pose = self.get_object_pose_w(object_key)
        target_place_pose = self.get_object_pose_w(target_place_key)
        
        # 2. 规划到物体上方 (Pre-grasp)
        pre_grasp_pose = obj_pose.clone()
        pre_grasp_pose[:, 2] += 0.15 
        success_pre_grasp_pose = self._plan_and_move(pre_grasp_pose, gripper_open=1.0, num_samples=10)
        # if success_pre_grasp_pose is not None:
            # pre_grasp_pose.copy_(success_pre_grasp_pose)
            # obj_pose.copy_(success_pre_grasp_pose)
        # self._wait_steps(gripper_val=1.0, steps=25) # 等待控制稳定
        

        # 3. 规划到抓取点并闭合
        obj_pose[:, :3] = self.scene["ee_frame"].data.target_pos_w[:, 0, :]
        obj_pose[:, 3:7] = self.scene["ee_frame"].data.target_quat_w[:, 0, :]
        obj_pose[:, 2] -= 0.15  # 更新为实际抓取位姿
        constraint_vec = [1, 1, 1, 0, 1, 1]  # 保持位置，允许绕 Z 轴旋转
        self._plan_and_move(obj_pose, gripper_open=1.0, constraint_pose=constraint_vec)
        self._wait_steps(gripper_val=-1.0, steps=25) # 闭合夹爪并等待物理触觉稳定

        # 4. 提起物体
        pre_grasp_pose[:, :3] = self.scene["ee_frame"].data.target_pos_w[:, 0, :]
        pre_grasp_pose[:, 3:7] = self.scene["ee_frame"].data.target_quat_w[:, 0, :]
        pre_grasp_pose[:, 2] += 0.15  # 更新为实际抓取位姿
        constraint_vec = [1, 1, 1, 0, 1, 1]  # 保持位置，允许绕 Z 轴旋转
        self._plan_and_move(pre_grasp_pose, gripper_open=-1.0, constraint_pose=constraint_vec)

        # 5. 移动到目标放置点上方
        pre_place_pose = target_place_pose.clone()
        pre_place_pose[:, 2] += 0.2
        success_pre_place_pose = self._plan_and_move(pre_place_pose, gripper_open=-1.0, num_samples=10)
        # if success_pre_place_pose is not None:
        #     pre_place_pose.copy_(success_pre_place_pose)
        #     target_place_pose.copy_(success_pre_place_pose)
        #     target_place_pose[:, 2] -= 0.05  # 更新为实际放置位姿
        
        # self._wait_steps(gripper_val=-1.0, steps=50) # 等待控制稳定

        # 6. 放置并释放
        target_place_pose[:, :3] = self.scene["ee_frame"].data.target_pos_w[:, 0, :]
        target_place_pose[:, 3:7] = self.scene["ee_frame"].data.target_quat_w[:, 0, :]
        target_place_pose[:, 2] -= 0.1  # 更新为实际抓取位姿
        constraint_vec = [0, 0, 0, 0, 1, 1]  # 保持位置，允许绕 Z 轴旋转
        self._plan_and_move(target_place_pose, gripper_open=-1.0, constraint_pose=constraint_vec)
        self._wait_steps(gripper_val=1.0, steps=25) 
        
        # 7. 撤回
        pre_place_pose[:, :3] = self.scene["ee_frame"].data.target_pos_w[:, 0, :]
        pre_place_pose[:, 3:7] = self.scene["ee_frame"].data.target_quat_w[:, 0, :]
        pre_place_pose[:, 2] += 0.1  # 更新为实际抓取位姿
        constraint_vec = [0, 0, 0, 0, 1, 1]  # 保持位置，允许绕 Z 轴旋转
        self._plan_and_move(pre_place_pose, gripper_open=1.0, constraint_pose=constraint_vec)
        
        self.recorder.stop_recording()

        if self.is_success.any():
            self.recorder.save_data()
            print(f"--- {object_key} 任务完成，数据已保存 ---")
        else:
            self.recorder.clear_data()
            print(f"--- {object_key} 任务未成功，未保存数据 ---")

    def _plan_and_move(
            self, 
            target_pose_w: torch.Tensor, 
            gripper_open: float, 
            constraint_pose=None, 
            num_samples: int = 1, 
            sample_axis: Tuple[float, float, float] = (1, 0, 0),
        ):
        """调用 Curobo 规划并执行"""
        if num_samples > 1:
            delta_quat = math_utils.quat_from_angle_axis(
                torch.arange(0., 2. * np.pi, 2. * np.pi / num_samples, device=target_pose_w.device, dtype=torch.float32),
                torch.tensor([sample_axis], device=target_pose_w.device, dtype=torch.float32),
            ).unsqueeze(0).repeat(target_pose_w.shape[0], 1, 1)
            candidate_poses = math_utils.combine_frame_transforms(
                target_pose_w[:, :3].unsqueeze(1).repeat(1, num_samples, 1).view(-1, 3),
                target_pose_w[:, 3:].unsqueeze(1).repeat(1, num_samples, 1).view(-1, 4),
                q12=delta_quat.view(-1, 4)
            )
            candidate_poses = torch.cat(candidate_poses, dim=1).view(-1, num_samples, 7)[:, 1:, :]
        else:
            candidate_poses = None


        if self.target_pose_visualizer is not None:
            try:
                visualize_poses = torch.cat((target_pose_w.unsqueeze(1), candidate_poses), dim=1) if candidate_poses is not None else target_pose_w.unsqueeze(1)
                for _ in range(2):
                    for i in range(visualize_poses.shape[1]):
                        translations = visualize_poses[:, i, :3].detach().cpu()
                        orientations = visualize_poses[:, i, 3:7].detach().cpu()
                        self.target_pose_visualizer.visualize(translations=translations, orientations=orientations)
                        for _ in range(10):
                            self.unwrapped_env.sim.step()
            except Exception:
                # ignore visualization errors to avoid breaking the main loop
                pass
        
        # 调用规划器 (确保 planner.py 接收 Tensor 并返回 Tensor)
        # 注意：此处 target_pose_w 是 [x,y,z, qw,qx,qy,qz]
        traj, success_target_pose = self.action_controller.plan(
            target_pose_w, constraint_pose=constraint_pose, candidate_target_poses=candidate_poses)

        if traj is not None:
            # 确保 traj 是 CUDA Tensor (N, 6)
            if not isinstance(traj, torch.Tensor):
                traj = torch.tensor(traj, device=self.device)
            self.execute_trajectory(traj, gripper_open)
        else:
            print(f"Warning: 规划到 {target_pose_w[:3]} 失败!")
        return success_target_pose

    def _wait_steps(self, gripper_val: float, steps: int = 10):
        """保持当前关节姿态，仅改变夹爪状态"""
        joint_ids = self.unwrapped_env.action_manager.get_term("arm_action")._joint_ids
        current_joints = self.robot.data.joint_pos[0, joint_ids].unsqueeze(0)
        gripper_tensor = torch.full((1, 1), gripper_val, device=self.device)
        action_vec = torch.cat([current_joints, gripper_tensor], dim=-1)
        
        for _ in range(steps):
            obs, reward, terminated, truncated, info = self.env.step(action_vec)
            self._record_step(action_vec, obs, reward, info)

    def close(self):
        self.env.close()

if __name__ == "__main__":
    runner = PickPlaceRunner()
    
    # 目标放置位姿 (CUDA Tensor: [x, y, z, qw, qx, qy, qz])
    target_pose = torch.tensor([[0.45, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]], device="cuda")
    
    try:
        while True:
            # 重置环境
            runner.reset()
            
            # 执行第一个物体的抓放
            runner.run_pick_place("high_object", "plate")
        
    # except KeyboardInterrupt:
    #     print("用户中断运行")
    except Exception as e:
        raise e
    # finally:
    #     runner.close()
    #     simulation_app.close()
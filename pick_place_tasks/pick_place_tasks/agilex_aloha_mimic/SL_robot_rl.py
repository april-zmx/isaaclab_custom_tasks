import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pick and lift state machine for lift environments.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--is_visual", type=bool, default=False, help="Visualize the grasp pose(True or Pose).")
parser.add_argument("--dataset_file", type=str, default="./datasets/SL_RL.hdf5", help="File path to export recorded demos.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import sys
import carb
import torch
import yaml
import copy
import numpy as np
import gymnasium as gym
from scipy.spatial.transform import Rotation as R

## ISAACLAB
import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import DatasetExportMode
from isaaclab.assets import Articulation, RigidObject
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData
from isaaclab.utils.math import combine_frame_transforms,subtract_frame_transforms
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.utils.math import quat_from_angle_axis, quat_apply
##curobo
from curobo.types.math import Pose
from curobo.types.base import TensorDeviceType
from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers

## self_defined
sys.path.append(os.getcwd())
import pick_place_tasks
from pick_place_tasks.agilex_aloha_mimic.action import Action

class Pose:
    def __init__(self,p=[0, -0.65, 0],q=[0.707, 0, 0, 0.707]):
        # 初始化p和q属性
        self.p = p
        self.q = q

def generate_random_quaternions(num=1, y_min=25, y_max=70, z_min=-30, z_max=30):
    """
    生成随机四元数（x=-180度，y/z∈[0°,90°]）
    输出格式：torch.Tensor [num, 4]（顺序w,x,y,z）
    """
    # 固定参数
    x_deg = -35 # x轴固定-180度

    # 生成随机y/z角度（弧度制）
    y_rand = np.random.uniform(y_min, y_max, num)
    z_rand = np.random.uniform(z_min, z_max, num)

    # 初始化四元数列表
    quat_list = []

    for y_deg, z_deg in zip(y_rand, z_rand):
        # 构造欧拉角（x,y,z顺序，角度制）
        euler_angles = [x_deg, y_deg, z_deg]
        
        # 转换为四元数（scipy输出顺序：x,y,z,w）
        rot = R.from_euler('xyz', euler_angles, degrees=True)
        quat_xyzw = rot.as_quat()  # [x,y,z,w]
        
        # 调整为w,x,y,z顺序
        quat_wxyz = np.roll(quat_xyzw, 1)  # 将w分量移到第一个位置
        
        # 转换为torch张量并添加到列表
        quat_list.append(torch.tensor(quat_wxyz, dtype=torch.float32))

    # 合并为二维张量
    return torch.stack(quat_list)

def main():
    # parse configuration
    env_cfg = parse_env_cfg(
        "Isaac-Agilex-PickPlace-ThreeObjects-v0", 
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric)
    
    # Initialize a visualization marker for target gripper poses
    try:
        marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/World/Visuals/target_pose")
        marker_cfg.markers["frame"].scale = (0.06, 0.06, 0.06)
        target_pose_visualizer = VisualizationMarkers(marker_cfg)
    except Exception:
        target_pose_visualizer = None

    # create environment
    env = gym.make("Isaac-Agilex-PickPlace-ThreeObjects-v0", cfg=env_cfg)
    env.reset()
    
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    action = Action(env.unwrapped.scene, sim)

    place_goal_left = np.array([-0.4,0.0,0.93,-0.707, 0, 0, -0.707])
    place_goal_right = np.array([0.4,0.0,1.0,-0.707, 0, 0, -0.707])
    count = 0
    use_arm = 'left_arm'
    while simulation_app.is_running():
        action.start_recording()

        # breakpoint()

        object_frame_sensor = env.unwrapped.scene["medium_object"]
        #
        object_position_w = object_frame_sensor.data.root_pos_w.clone().cpu()
        object_orientation_w = object_frame_sensor.data.root_quat_w.clone().cpu()
        # object_quat_w = generate_random_quaternions(y_min=25,y_max=70,z_min=50,z_max=80).cpu()
        
        object_orientation_w = quat_from_angle_axis(
            torch.tensor([np.pi / 2.] * env.num_envs, device=env.device, dtype=torch.float32),
            torch.tensor([[0., 0., 1.]] * env.num_envs, device=env.device, dtype=torch.float32),
        ).cpu()

        target_gripper_pose = torch.cat((object_position_w, object_orientation_w), dim=1)

        # Visualize the target gripper pose (translations: (M,3), orientations: (M,4:wxyz))
        if target_pose_visualizer is not None:
            try:
                translations = target_gripper_pose[:, :3].detach().cpu()
                orientations = target_gripper_pose[:, 3:7].detach().cpu()
                target_pose_visualizer.visualize(translations=translations, orientations=orientations)
            except Exception:
                # ignore visualization errors to avoid breaking the main loop
                pass
        
        action.take_grasp_action(target_gripper_pose=target_gripper_pose, arm_tag=use_arm)

        # 放置
        object_frame_sensor = env.unwrapped.scene["plate"]
        object_position_w = object_frame_sensor.data.root_pos_w.clone().cpu()
        object_orientation_w = object_frame_sensor.data.root_quat_w.clone().cpu()
        object_position_w[:, 2] += 0.1
        object_orientation_w = quat_from_angle_axis(
            torch.tensor([np.pi / 2.] * env.num_envs, device=env.device, dtype=torch.float32),
            torch.tensor([[0., 0., 1.]] * env.num_envs, device=env.device, dtype=torch.float32),
        ).cpu()
        target_place_pose = torch.cat((object_position_w, object_orientation_w), dim=1)

        # Visualize the target gripper pose (translations: (M,3), orientations: (M,4:wxyz))
        if target_pose_visualizer is not None:
            try:
                translations = target_place_pose[:, :3].detach().cpu()
                orientations = target_place_pose[:, 3:7].detach().cpu()
                target_pose_visualizer.visualize(translations=translations, orientations=orientations)
            except Exception:
                # ignore visualization errors to avoid breaking the main loop
                pass
        
        for _ in range(10):
            sim.step()
        
        # breakpoint()
        action.take_place_action(target_place_pose=target_place_pose, arm_tag=use_arm)
        action.stop_recording()

        object_frame_sensor = env.unwrapped.scene["medium_object"]
        object_position_w = object_frame_sensor.data.root_pos_w[..., 0, :].clone().squeeze(0)
        if(object_position_w[2]>0.82):
            action.save_recording()
            action.clear_recording()
            count += 1
            if(count>=500):
                break
        else:
            action.clear_recording()
        env.reset()

    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

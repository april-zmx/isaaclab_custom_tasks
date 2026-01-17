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
import omni.replicator.core as rep 
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
##curobo
from curobo.types.math import Pose
from curobo.types.base import TensorDeviceType

## self_defined
sys.path.append(os.getcwd())
import IFlytek
from IFlytek.Tools import State_Machine
from IFlytek.Thrid_pkg.Utils import Curobo_Utils
from IFlytek.Robot_Setups.sl_lift.config.robot import *  
from IFlytek.Robot_Setups.sl_lift.lift_env_cfg import LiftEnvCfg
from IFlytek.Thrid_pkg.Utils.GraspPredictor import GraspPredictor
from IFlytek.Tools.Usd_Help import visualize_pose_marker, get_obstacle_collision_world
from IFlytek.Tools.Get_Robot_State import get_current_joint_state
from IFlytek.Tools.Image_Deal import capture_and_predict_grasps,expand_white_area,set_object_semantic_tags,transform_grasp_pose
from IFlytek.Scripts.SL.planner import CuroboPlanner
from IFlytek.Scripts.SL.action import Action
from IFlytek.Scripts.SL.recorder import RobotDataRecorder

class Pose:
    def __init__(self,p=[0, -0.65, 0],q=[0.707, 0, 0, 0.707]):
        # 初始化p和q属性
        self.p = p
        self.q = q
class Planner():
    def __init__(self,scene):

        left_entity_origion_pose = Pose()
        right_entity_origion_pose = Pose()
        left_arm_joints_name = ['fl_joint1','fl_joint2','fl_joint3','fl_joint4','fl_joint5','fl_joint6']
        right_arm_joints_name = ['fr_joint1','fr_joint2','fr_joint3','fr_joint4','fr_joint5','fr_joint6']
        joint_all = scene['robot'].joint_names
        #导入配置文件 
        abs_left_curobo_yml_path = '/home/iflytek/IsaacLab2/IFlytek/Assets/Robots/RoboTwin/assets/embodiments/aloha-agilex/curobo_left.yml'
        abs_right_curobo_yml_path = '/home/iflytek/IsaacLab2/IFlytek/Assets/Robots/RoboTwin/assets/embodiments/aloha-agilex/curobo_right.yml'
        config_path = '/home/iflytek/IsaacLab2/IFlytek/Assets/Robots/RoboTwin/assets/embodiments/aloha-agilex/config.yml'
        with open(config_path, "r") as f:
            robot_yml_data = yaml.safe_load(f)
        self.left_gripper_bias = robot_yml_data['gripper_bias']
        self.right_gripper_bias = robot_yml_data['gripper_bias']  
        self.left_delta_matrix = np.array(robot_yml_data['delta_matrix'])
        self.left_inv_delta_matrix = np.linalg.inv(self.left_delta_matrix)
        self.right_delta_matrix = np.array(robot_yml_data['delta_matrix'])
        self.right_inv_delta_matrix = np.linalg.inv(self.left_delta_matrix)
        # 初始化规划器
        self.left_planner = CuroboPlanner(left_entity_origion_pose,
                                    left_arm_joints_name,
                                    joint_all,
                                    yml_path=abs_left_curobo_yml_path)
        self.right_planner = CuroboPlanner(right_entity_origion_pose,
                                            right_arm_joints_name,
                                            joint_all,
                                            yml_path=abs_right_curobo_yml_path)
        
def set_object_semantic_tags(path_pattern: str, semantic_tags: list[tuple[str, str]]):
    """
    设置指定物体的语义标签（用于Omniverse Replicator）
    
    参数:
    path_pattern: 物体基元的路径模式（如 "/World/envs/env_0/Kiwi"）
    semantic_tags: 要设置的语义标签列表，格式为 [(key1, value1), (key2, value2), ...]
    
    返回:
    rep.utils.PrimPath: 获取到的物体基元对象
    """
    # 获取指定路径的物体基元
    object_prims = rep.get.prims(path_pattern=path_pattern)
    # 设置语义标签
    with object_prims:
        rep.modify.semantics(semantic_tags)
    return object_prims
def generate_random_quaternions(num=1, y_min=25, y_max=70, z_min=-30, z_max=30):
    """
    生成随机四元数（x=-180度，y/z∈[0°,90°]）
    输出格式：torch.Tensor [num, 4]（顺序w,x,y,z）
    """
    # 固定参数
    x_deg = 0# x轴固定-180度

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
    env_cfg: LiftEnvCfg = parse_env_cfg("Isaac-sl-robot-rl-control", device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)


    # create environment
    env = gym.make("Isaac-sl-robot-rl-control", cfg=env_cfg)
    env.reset()
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    # create action buffers (arm angle + gripper)
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    planner = Planner(env.unwrapped.scene)
    left_planner = planner.left_planner
    right_planner = planner.right_planner

    action = Action(env.unwrapped.scene,sim)

    select_object = "/World/envs/env_0/Bottle"
    object_mask = set_object_semantic_tags(select_object, [('class', 'object')])

    place_goal_left = np.array([-0.4,0.0,1.0,-0.707, 0, 0, -0.707])#(XYZ,WZYZ)
    place_goal_right = np.array([0.4,0.0,1.0,-0.707, 0, 0, -0.707])
    count = 0
    while simulation_app.is_running():
        action.start_recording()

        robot: Articulation = env.unwrapped.scene["robot"]
        object_data: RigidObjectData = env.unwrapped.scene["bottled"].data
        object_position_w = object_data.root_pos_w[..., 0, :].clone().squeeze(0)
        object_position_w[2] += 0.1
        object_orientation_w = object_data.root_quat_w[..., 0, :].clone().squeeze(0)
        ori = generate_random_quaternions(y_min=0,y_max=20,z_min=0,z_max=70).cpu().numpy().squeeze()
        ee_goal = np.concatenate([object_position_w.cpu().numpy(), ori])
        # joint_pos = copy.deepcopy(env.unwrapped.scene['robot'].data.joint_pos.cpu().squeeze().numpy())
        use_arm = 'left_arm'
        action.take_grasp_action(pos=ee_goal,arm_tag=use_arm)

        if(use_arm=="left_arm"):
            action.take_place_action(pos=place_goal_left,arm_tag=use_arm)
        else:
            action.take_place_action(pos=place_goal_right,arm_tag=use_arm)
        action.stop_recording()

        object_frame_sensor = env.unwrapped.scene["bottled"]
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

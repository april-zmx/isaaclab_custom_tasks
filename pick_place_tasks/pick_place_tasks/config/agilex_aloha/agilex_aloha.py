import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import os

CUSTOM_USD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")

##
# Configuration
##

AGILEX_ALOHA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(CUSTOM_USD_DIR, "arx5_description_isaac/arx5_description_isaac.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    # 初始状态配置
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # # 为主要关节设置初始位置（0表示默认零位）
            # "fl_joint1": 0.0, "fr_joint1": 0.0,
            # "lr_joint1": 0.0, "rr_joint1": 0.0,
            # "fl_joint2": 0.0, "fr_joint2": 0.0,
            # "lr_joint2": 0.0, "rr_joint2": 0.0,
            ".*": 0.0,
        },
        # 机器人初始位置
        pos=(0.0, -0.55, 0.0),
        rot=(0.707,0,0,0.707),
    ),
    actuators={
        # 1号关节组（所有前缀的joint1）
        "joint1_actuators": ImplicitActuatorCfg(
            joint_names_expr=["[a-z]+_joint1"],
            effort_limit_sim=100.0,
            velocity_limit_sim=3.0,
            stiffness=80.0,
            damping=15.0,
        ),
        # 2号关节组（所有前缀的joint2）
        "joint2_actuators": ImplicitActuatorCfg(
            joint_names_expr=["[a-z]+_joint2"],
            effort_limit_sim=100.0,
            velocity_limit_sim=3.0,
            stiffness=70.0,
            damping=15.0,
        ),
        # 3-4号关节组（所有前缀的joint3和joint4）
        "joint3_4_actuators": ImplicitActuatorCfg(
            joint_names_expr=["[a-z]+_joint[3-4]"],
            effort_limit_sim=100.0,
            velocity_limit_sim=3.0,
            stiffness=60.0,
            damping=10.0,
        ),
        # 5-6号关节组（所有前缀的joint5和joint6）
        "joint5_6_actuators": ImplicitActuatorCfg(
            joint_names_expr=["[a-z]+_joint[5-6]"],
            effort_limit_sim=100.0,
            velocity_limit_sim=3.0,
            stiffness=50.0,
            damping=5.0,
        ),
        # 7-8号关节组（所有前缀的joint7和joint8）
        "joint7_8_actuators": ImplicitActuatorCfg(
            joint_names_expr=["[a-z]+_joint[7-8]"],
            effort_limit_sim=10.0,
            velocity_limit_sim=1.0,
            stiffness=1000.0,
            damping=1.0,
        ),
        # 主轮组
        "main_wheels": ImplicitActuatorCfg(
            joint_names_expr=["fl_wheel", "fr_wheel", "left_wheel", "right_wheel", "rl_wheel", "rr_wheel"],
            effort_limit_sim=30.0,
            velocity_limit_sim=5.0,
            stiffness=3000.0,
            damping=30.0,
        ),
        # 万向轮组
        "castor_wheels": ImplicitActuatorCfg(
            joint_names_expr=["fl_castor_wheel", "fr_castor_wheel", "rl_castor_wheel", "rr_castor_wheel"],
            effort_limit_sim=20.0,
            velocity_limit_sim=5.0,
            stiffness=2000.0,
            damping=20.0,
        ),
    },
)
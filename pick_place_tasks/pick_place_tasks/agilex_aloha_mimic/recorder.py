import os
import cv2
import numpy as np
import pandas as pd
import time
from datetime import datetime

class RobotDataRecorder:
    """
    机器人数据记录器，用于记录多路图像、机械臂关节状态和动作信息，并支持保存为视频和CSV文件
    """
    def __init__(self, output_dir="recordings"):
        """
        初始化记录器
        
        参数:
            output_dir: 输出目录路径
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 数据缓冲区
        self.images = {}
        self.joint_states = []
        self.actions = []
        self.timestamps = []
        
        # 视频写入器
        self.video_writers = {}
        
        # 记录状态
        self.is_recording = False
        self.record_start_time = None
        self.record_end_time = None
        self.stop_reason = "USER_STOP"
    
    def start_recording(self):
        """
        开始记录数据
        """
        if not self.is_recording:
            self.is_recording = True
            self.record_start_time = time.time()
            print(f"开始记录数据，开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def stop_recording(self):
        """
        停止记录数据
        """
        if self.is_recording:
            self.is_recording = False
            self.record_end_time = time.time()
            self.stop_reason = "USER_STOP"
            print(f"停止记录数据，结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def record_image(self, camera_name, image):
        """
        记录单路相机图像
        
        参数:
            camera_name: 相机名称，用于区分多路图像
            image: 图像数据 (numpy array)
        """
        if not self.is_recording:
            return
            
        if camera_name not in self.images:
            self.images[camera_name] = []
            
        self.images[camera_name].append(image.copy())
    
    def record_joint_state(self, joint_state):
        """
        记录机械臂关节状态
        
        参数:
            joint_state: 关节状态，可以是字典、列表或numpy数组
        """
        if not self.is_recording:
            return
            
        # 确保是字典格式
        if isinstance(joint_state, (list, np.ndarray)):
            joint_state = {f'joint_{i}': value for i, value in enumerate(joint_state)}
        
        self.joint_states.append(joint_state.copy())
    
    def record_action(self, action):
        """
        记录动作信息
        
        参数:
            action: 动作数据，可以是字典、列表或numpy数组
        """
        if not self.is_recording:
            return
            
        # 确保是字典格式
        if isinstance(action, (list, np.ndarray)):
            action = {f'action_{i}': value for i, value in enumerate(action)}
        
        self.actions.append(action.copy())
    
    def save_data(self, session_name=None):
        """
        保存记录的数据
        
        参数:
            session_name: 会话名称，用于生成文件夹名，如不提供则使用时间戳
        """
        if session_name is None:
            session_name = datetime.now().strftime("episode_"+'%Y-%m-%d_%H_%M_%S')
            
        session_dir = os.path.join(self.output_dir, session_name)
        os.makedirs(session_dir, exist_ok=True)
        
        # 保存视频
        self._save_videos(session_dir)
        
        # 保存CSV文件
        self._save_csv(session_dir)

        self._save_info_file(session_dir)
        
        print(f"数据已保存到: {session_dir}")
        return session_dir
    
    def _save_videos(self, session_dir):
        """
        将记录的图像保存为视频
        
        参数:
            session_dir: 会话目录
        """
        
        for camera_name, image_list in self.images.items():
            if len(image_list) == 0:
                continue
                
            # 获取图像尺寸
            height, width = image_list[0].shape[:2]
            
            # 创建视频写入器
            if camera_name == "left_camera":
                file_name = "cam_left"
                video_path = os.path.join(session_dir,file_name, f"{file_name}.mp4")
            elif camera_name == "right_camera":
                file_name = "cam_right"
                video_path = os.path.join(session_dir,file_name, f"{file_name}.mp4")
            elif camera_name == "head_camera":
                file_name = "cam_head"
                video_path = os.path.join(session_dir,file_name, f"{file_name}.mp4")
            else:
                file_name = camera_name
                video_path = os.path.join(session_dir,file_name, f"{file_name}.mp4")
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30  # 默认帧率，可以根据需要调整
            
            try:
                # 直接使用mp4v编码器，确保能够创建VideoWriter
                out = cv2.VideoWriter(video_path, fourcc, fps, (width, height), True)
                
                # 检查VideoWriter是否成功初始化
                if not out.isOpened():
                    # 如果失败，尝试其他常用编码器
                    print(f"警告: 无法使用mp4v编码器，尝试其他编码器")
                    # 尝试MJPG编码器
                    fourcc_mjpg = cv2.VideoWriter_fourcc(*'MJPG')
                    out = cv2.VideoWriter(video_path, fourcc_mjpg, fps, (width, height), True)
                    
                    if not out.isOpened():
                        # 尝试XVID编码器
                        fourcc_xvid = cv2.VideoWriter_fourcc(*'XVID')
                        out = cv2.VideoWriter(video_path, fourcc_xvid, fps, (width, height), True)
            except Exception as e:
                print(f"创建VideoWriter时出错: {e}")
                # 作为最后的尝试，使用不带颜色参数的构造函数
                out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            
            # 写入视频帧
            for image in image_list:
                # 确保图像是BGR格式（OpenCV要求）
                if image.dtype == bool:
                    # 将布尔类型转换为uint8（0和255）
                    image = image.astype(np.uint8) * 255

                if len(image.shape) == 2:  # 灰度图
                    if image.dtype != np.uint8:
                        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                elif image.shape[2] == 3:  # RGB图
                    # 确保图像深度是8位（OpenCV常用格式）
                    if image.dtype != np.uint8:
                        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # 确保图像尺寸与视频设置一致
                if image.shape[1] != width or image.shape[0] != height:
                    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
                
                out.write(image)
                
            # 释放资源
            out.release()
            print(f"视频已保存: {video_path}")
    
    def _save_csv(self, session_dir):
        """
        将关节状态和动作数据合并保存到一个CSV文件中
        
        参数:
            session_dir: 会话目录
        """
        csv_dir = session_dir
        
        # 创建合并的DataFrame
        merged_data = []
        
        # 获取最大数据长度
        max_length = max(len(self.joint_states), len(self.actions))
        
        for i in range(max_length):
            data_point = {}
            
            # 添加时间戳（如果存在）
            if i < len(self.timestamps):
                data_point['timestamp'] = self.timestamps[i]
            else:
                data_point['timestamp'] = None
            
            # 添加关节状态（如果存在）
            if i < len(self.joint_states):
                # 添加前缀"joint_"以区分关节状态，并根据规则转换关节名称
                for key, value in self.joint_states[i].items():
                    # 提取key中的前缀和数字
                    new_key = key
                    # 处理fl_joint*格式的key
                    if key.startswith('fl_joint'):
                        # 提取数字部分并转换为整数
                        joint_num_str = ''.join(filter(str.isdigit, key))
                        if joint_num_str:
                            joint_num = int(joint_num_str)
                            # 执行减法运算后转换回字符串
                            new_key = f'l_s_{joint_num - 1}'
                    # 处理fr_joint*格式的key
                    elif key.startswith('fr_joint'):
                        # 提取数字部分并转换为整数
                        joint_num_str = ''.join(filter(str.isdigit, key))
                        if joint_num_str:
                            joint_num = int(joint_num_str)
                            # 执行减法运算后转换回字符串
                            new_key = f'r_s_{joint_num - 1}'
                    
                    # 添加joint_前缀
                    data_point[f"{new_key}"] = value
            
            # 添加动作数据（如果存在）
            if i < len(self.actions):
                # 添加前缀"action_"以区分动作数据，并根据规则转换关节名称
                for key, value in self.actions[i].items():
                    # 提取key中的前缀和数字
                    new_key = key
                    # 处理fl_joint*格式的key
                    if key.startswith('fl_joint'):
                        # 提取数字部分并转换为整数
                        joint_num_str = ''.join(filter(str.isdigit, key))
                        if joint_num_str:
                            joint_num = int(joint_num_str)
                            # 执行减法运算后转换回字符串
                            new_key = f'l_a_{joint_num - 1}'
                    # 处理fr_joint*格式的key
                    elif key.startswith('fr_joint'):
                        # 提取数字部分并转换为整数
                        joint_num_str = ''.join(filter(str.isdigit, key))
                        if joint_num_str:
                            joint_num = int(joint_num_str)
                            # 执行减法运算后转换回字符串
                            new_key = f'r_a_{joint_num - 1}'
                    
                    # 添加action_前缀
                    data_point[f"{new_key}"] = value
            
            merged_data.append(data_point)
        
        # 保存合并后的数据
        if merged_data:
            merged_df = pd.DataFrame(merged_data)
            
            # 确保时间戳是第一列
            if 'timestamp' in merged_df.columns:
                cols = ['timestamp'] + [col for col in merged_df.columns if col != 'timestamp']
                merged_df = merged_df[cols]
            
            merged_csv_path = os.path.join(csv_dir, "robot_data.csv")
            merged_df.to_csv(merged_csv_path, index=False)
            print(f"合并数据已保存: {merged_csv_path}")
    
    def clear_data(self):
        """
        清除所有记录的数据
        """
        self.images = {}
        self.joint_states = []
        self.actions = []
        self.timestamps = []
        print("数据已清除")
    
    def get_recorded_info(self):
        """
        获取记录信息
        
        返回:
            字典，包含记录的统计信息
        """
        info = {
            "is_recording": self.is_recording,
            "camera_count": len(self.images),
            "joint_state_count": len(self.joint_states),
            "action_count": len(self.actions)
        }
        
        if self.is_recording and self.record_start_time is not None:
            info["recording_time"] = time.time() - self.record_start_time
            
        # 相机图像数量
        for camera_name, image_list in self.images.items():
            info[f"{camera_name}_frames"] = len(image_list)
            
        return info
    
    def _save_info_file(self, session_dir):
        """
        保存记录信息到info.txt文件
        
        参数:
            session_dir: 会话目录
        """
        # 计算总帧数（取所有相机中最大的帧数）
        total_frames = 0
        for camera_name, image_list in self.images.items():
            total_frames = max(total_frames, len(image_list))
        
        # 格式化时间
        start_time_str = "N/A"
        end_time_str = "N/A"
        if self.record_start_time is not None:
            start_time_str = datetime.fromtimestamp(self.record_start_time).strftime('%Y-%m-%d %H:%M:%S')
        if self.record_end_time is not None:
            end_time_str = datetime.fromtimestamp(self.record_end_time).strftime('%Y-%m-%d %H:%M:%S')
        
        # 生成info.txt内容
        info_content = (
            f"StopReason = {self.stop_reason}\n"
            f"总帧数 = {total_frames}\n"
            f"开始记录时间 = {start_time_str}\n"
            f"终止时间 = {end_time_str}\n"
        )
        
        # 保存到文件
        info_file_path = os.path.join(session_dir, "info.txt")
        with open(info_file_path, 'w', encoding='utf-8') as f:
            f.write(info_content)
        
        print(f"信息文件已保存: {info_file_path}")
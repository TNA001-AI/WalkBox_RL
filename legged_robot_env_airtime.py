# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# test1
from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from .legged_robot_cfg import LEGGED_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from omni.isaac.lab.scene import InteractiveSceneCfg

from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg
# from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg, save_images_to_file
# from omni.isaac.lab.sensors import ImuCfg, Imu, patterns, ImuData

from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform

@configclass
class RobotImuEnvCfg(DirectRLEnvCfg):
    # env
    #TODO
    decimation = 2
    episode_length_s = 60.0  
    action_scale = 100.0  # [N]


    # state and action spaces
    # TODO
    action_space = 6  
    observation_space = 18  
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = LEGGED_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    terrain = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="plane",
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="average",
        restitution_combine_mode="average",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    ),
    debug_vis=False,
)
    # sensors
    # imu = ImuCfg(
    #     prim_path="/World/Robot/base_link",
    #     update_period=0.1,
    #     offset=ImuCfg.OffsetCfg(
    #         pos=(0.0, 0.0, 0.1),
    #         rot=(1.0, 0.0, 0.0, 0.0),
    #     ),
    # gravity_bias=(0.0, 0.0, 9.81),
    # )

    contact_forces_LF = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/WalkBox3/WalkBox/Link0_2",
        track_pose=True,
        debug_vis=False,
        update_period=0.0,
        track_air_time=True,
        history_length=3,
        filter_prim_paths_expr=["/World/ground"],
    )

    contact_forces_RF = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/WalkBox3/WalkBox/Link1_2",
        track_pose=True,
        debug_vis=False,
        update_period=0.0,
        track_air_time=True,
        history_length=3,
        filter_prim_paths_expr=["/World/ground"],
    )

    write_image_to_file = False



    # change viewer settings
    viewer = ViewerCfg(eye=(20.0, 20.0, 20.0))

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=20.0, replicate_physics=True)

    # reward scales
    # TODO
    rew_scale_alive = 0.5
    rew_scale_terminated = -20.0

    rew_scale_dist = -0.05

    rew_scale_direction = -0.5

    rew_scale_head_velocity = 1.2 # 1

    rew_scale_dof_vel = 1

    rew_scale_airtime = 1 # 1





class RobotImuEnv(DirectRLEnv):

    cfg: RobotImuEnvCfg

    def __init__(
        self, cfg: RobotImuEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        # self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        # self.robot_dof_speed_scales[:] = 0.1

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        self.joint_pos = self._robot.data.joint_pos
        self.joint_vel = self._robot.data.joint_vel

        # joints' idx
        self.left_hip_joint_idx    = self._robot.find_joints("Joint0_0")
        self.left_knee_joint_idx   = self._robot.find_joints("Joint0_1")
        self.left_ankle_joint_idx  = self._robot.find_joints("Joint0_2")
        self.right_hip_joint_idx   = self._robot.find_joints("Joint1_0") 
        self.right_knee_joint_idx  = self._robot.find_joints("Joint1_1")
        self.right_ankle_joint_idx = self._robot.find_joints("Joint1_2")

        # head link idx
        self.head_link_idx = self._robot.find_bodies("base_link")[0][0]
        self.left_foot_idx = self._robot.find_bodies("Link0_2")[0][0]
        self.right_foot_idx = self._robot.find_bodies("Link1_2")[0][0]
        self.head_pos = self._robot.data.body_pos_w[:, self.head_link_idx]
        # self.target = torch.tensor([1000, 0, 4], dtype=torch.float32, device=self.sim.device).repeat(
        #     (self.num_envs, 1)
        # )

        # 用于记录左右脚的离地状态
        self.left_foot_airtime = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.right_foot_airtime = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)


        # target
        self.target =  self.head_pos.clone()
        self.target[:,0] += 10

        # print("head_pos:",self.head_pos)
        # print("target:",self.target)
        self.reset_terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def close(self):
        """Cleanup for the environment."""
        super().close()

    def _setup_scene(self):
        """Setup the scene."""
        self._robot = Articulation(self.cfg.robot_cfg)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # self._imu = Imu(self.cfg.imu)
        # clone, filter, and replicate
        self._contact_sensor0: ContactSensor = ContactSensor(self.cfg.contact_forces_LF)
        self._contact_sensor1: ContactSensor = ContactSensor(self.cfg.contact_forces_RF)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        # add articulation and sensors to scene
        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["contact_sensor0"] = self._contact_sensor0
        self.scene.sensors["contact_sensor1"] = self._contact_sensor1
        # self.scene.sensors["imu"] = self._imu
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        # print("joint_pos:", self._robot.data.joint_pos, "shape:", self._robot.data.joint_pos.shape)
        # print("robot_dof_lower_limits:", self.robot_dof_lower_limits, "shape:", self.robot_dof_lower_limits.shape)       
        self.actions = actions.clone().clamp(-1.0, 1.0)
        dof_targets = self.robot_dof_targets + self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(dof_targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        # print("robot_dof_targets:", self.robot_dof_targets)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    def _get_observations(self, env_ids: torch.Tensor | None = None) -> dict:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        # print("joint_pos:", self._robot.data.joint_pos, "shape:", self._robot.data.joint_pos.shape)
        # print("robot_dof_lower_limits:", self.robot_dof_lower_limits, "shape:", self.robot_dof_lower_limits.shape)       
        # # print(self.robot_dof_upper_limits - self.robot_dof_lower_limits)

        # dof_pos_scaled = (
        #     2.0
        #     * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
        #     / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
        #     - 1.0
        # )        
        dof_pos_scaled = self._robot.data.joint_pos
        self.head_pos = self._robot.data.body_pos_w[env_ids, self.head_link_idx]
        self.head_rot = self._robot.data.body_quat_w[env_ids, self.head_link_idx]
        # linear and angular velocities
        self.head_vel = self._robot.data.body_vel_w[env_ids, self.head_link_idx]

        self.to_target = torch.norm(self.head_pos - self.target[env_ids], p=2, dim=-1).unsqueeze(-1)
        # PAN
        # print("to_target:",self.to_target)

        # # Debugging: Print values and shapes
        # print("dof_pos_scaled:", dof_pos_scaled, "shape:", dof_pos_scaled.shape)
        # print("self._robot.data.joint_vel:", self._robot.data.joint_vel, "shape:", self._robot.data.joint_vel.shape)
        # print("self.to_target:", self.to_target, "shape:", self.to_target.shape)
        # print("head_pos:", self.head_pos, "shape:", self.head_pos.shape)
        # print("head_rot:", self.head_rot, "shape:", self.head_rot.shape)
        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel,
                self.to_target,
                self.head_pos,
                self.head_rot,
                self.head_vel
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    def _get_rewards(self) -> torch.Tensor:
        self.head_pos = self._robot.data.body_pos_w[:, self.head_link_idx]
        self.head_rot = self._robot.data.body_quat_w[:, self.head_link_idx]
        # linear and angular velocities
        self.head_vel = self._robot.data.body_lin_vel_w[:, self.head_link_idx]
        # print("head_rot:", self.head_rot, "shape:", self.head_rot.shape)
        dof_vel = self._robot.data.joint_vel
        total_reward = self.compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_dist,
            self.cfg.rew_scale_direction,
            self.cfg.rew_scale_head_velocity,
            self.cfg.rew_scale_dof_vel,
            self.cfg.rew_scale_dof_vel,
            self.head_pos,
            self.head_rot,
            self.target, 
            self.head_vel,
            dof_vel,
            # self.joint_pos[:, self.left_hip_joint_idx[0]],
            # self.joint_vel[:, self.left_hip_joint_idx[0]],
            self.reset_terminated,
        )

        return total_reward



    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.all_head_pos = self._robot.data.body_pos_w[:, self.head_link_idx]
        
        self.to_target = torch.norm(self.all_head_pos - self.target, p=2, dim=-1)
        
        ##判断高度        
        # # 地面高度阈值
        # ground_level = 0.09  # 地面高度 6cm
    
        # 获取脚部 z 坐标
        # left_foot_pos = self._robot.data.body_pos_w[:, self.left_foot_idx, 2]
        # right_foot_pos = self._robot.data.body_pos_w[:, self.right_foot_idx, 2]
        # print("left_foot_pos:",left_foot_pos)
        # print("right_foot_pos:",right_foot_pos)
        # # 检测是否离地
        # left_foot_in_air = left_foot_pos > ground_level
        # right_foot_in_air = right_foot_pos > ground_level

        ## 判断角度
        # # 获取脚部四元数
        # left_foot_quaternion = self._robot.data.body_quat_w[:, self.left_foot_idx]
        # right_foot_quaternion = self._robot.data.body_quat_w[:, self.right_foot_idx]

        # # 设置脚部姿态偏离地面的角度阈值（单位：rad）
        # orientation_threshold = 15.0 * (torch.pi / 180)

        # # 定义函数：将四元数转为欧拉角
        # def quaternion_to_euler(q):
        #     """
        #     输入: 四元数 q (形状: [..., 4])
        #     返回: 欧拉角 (roll, pitch, yaw) 形状: [..., 3]
        #     """
        #     x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        #     t0 = +2.0 * (w * x + y * z)
        #     t1 = +1.0 - 2.0 * (x * x + y * y)
        #     roll = torch.atan2(t0, t1)  # 横滚角（roll）

        #     t2 = +2.0 * (w * y - z * x)
        #     t2 = torch.clamp(t2, -1.0, +1.0)
        #     pitch = torch.asin(t2)  # 俯仰角（pitch）

        #     t3 = +2.0 * (w * z + x * y)
        #     t4 = +1.0 - 2.0 * (y * y + z * z)
        #     yaw = torch.atan2(t3, t4)  # 偏航角（yaw）

        #     return torch.stack((roll, pitch, yaw), dim=-1)

        # # 转换为欧拉角
        # left_foot_euler = quaternion_to_euler(left_foot_quaternion)
        # right_foot_euler = quaternion_to_euler(right_foot_quaternion)

        # # 计算俯仰角和横滚角的绝对值
        # left_foot_angle_offset = torch.abs(left_foot_euler[:, 0]) + torch.abs(left_foot_euler[:, 1])
        # right_foot_angle_offset = torch.abs(right_foot_euler[:, 0]) + torch.abs(right_foot_euler[:, 1])

        # # 判断是否在姿态阈值内
        # left_foot_valid = left_foot_angle_offset < orientation_threshold
        # right_foot_valid = right_foot_angle_offset < orientation_threshold

        # # 检测是否离地
        # left_foot_in_air = (left_foot_pos > ground_level) & left_foot_valid
        # right_foot_in_air = (right_foot_pos > ground_level) & right_foot_valid
        # 更新离地时间
        # self.left_foot_airtime[left_foot_in_air] += self.dt
        # self.right_foot_airtime[right_foot_in_air] += self.dt
        # self.left_foot_airtime[~left_foot_in_air] = 0
        # self.right_foot_airtime[~right_foot_in_air] = 0

        self.left_foot_airtime = self._contact_sensor0.data.current_air_time.squeeze(-1)
        self.right_foot_airtime = self._contact_sensor1.data.current_air_time.squeeze(-1)
        # print("left_foot_airtime",self.left_foot_airtime, "shape:", self.left_foot_airtime.shape)
        # if the robot has reached the target
        reached_target = self.to_target < 0.5

        # if the head's z-position is less than 0.15 units
        fallen = self.all_head_pos[:, 2] < 0.20

        # Combine termination conditions
        terminated = reached_target | fallen
        
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        # print('max_episode_length:',self.max_episode_length)
        self.reset_terminated = terminated  # Update the reset flag
        
        
        return terminated, truncated


    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)
        # robot state
        # joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
        #     -0.125,
        #     0.125,
        #     (len(env_ids), self._robot.num_joints),
        #     self.device,
        # )
        # robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.0125,
            0.0125,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )
        # joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)

        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values(env_ids)

    # auxiliary methods

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        self.head_pos = self._robot.data.body_pos_w[env_ids, self.head_link_idx]
        self.head_rot = self._robot.data.body_quat_w[env_ids, self.head_link_idx]
        
        # print("id:", env_ids,"head_pos:",self.head_pos,"shape:", self.head_pos.shape)

        self.to_target = torch.norm(self.head_pos - self.target[env_ids], p=2, dim=-1)
        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        # print("to_target:", self.target[env_ids],"shape:",self.target[env_ids].shape)

# @torch.jit.script
# def compute_rewards(
#     rew_scale_alive: float,
#     rew_scale_terminated: float,
#     rew_scale_dist:float,
#     head_pos: torch.Tensor,
#     targets:torch.Tensor,
#     reset_terminated: torch.Tensor,
# ):
#     rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
#     rew_termination = rew_scale_terminated * reset_terminated.float()
#     d = torch.norm(head_pos - targets, p=2, dim=-1)
#     dist_reward = 1.0 / (1.0 + d**2)
#     dist_reward *= dist_reward
#     rew_dist = rew_scale_dist * dist_reward

#     # rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
#     # rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
#     # rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
#     total_reward = rew_alive + rew_termination + rew_dist
#     return total_reward

#PAN
    def compute_rewards(self,
        rew_scale_alive: float,
        rew_scale_terminated: float,
        rew_scale_dist: float,
        rew_scale_direction: float,
        rew_scale_head_velocity: float,
        rew_scale_dof_vel: float,
        rew_scale_airtime:float,
        head_pos: torch.Tensor,
        head_rot: torch.Tensor,
        targets: torch.Tensor,
        head_vel: torch.Tensor,
        dof_vel: torch.Tensor,
        reset_terminated: torch.Tensor,
    ):
        # Compute distance-based reward: Euclidean distance
        d = torch.norm(head_pos - targets, p=2, dim=-1)
        # print("head_pos: ", head_pos)
        # print("d: ", head_pos - targets)
        # the more closer, the higher of dist_reward
        dist_reward = d
        rew_dist = rew_scale_dist * dist_reward
        rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
        rew_termination = rew_scale_terminated * reset_terminated.float()
        rew_direction = rew_scale_direction * quaternion_to_angle(head_rot)
        rew_vel = rew_scale_head_velocity * head_vel[:,0]
        
        # print("rew_dof_vel:",dof_vel)
        # hip_vel = (torch.abs(dof_vel[:, self.left_hip_joint_idx[0]]) + 
        #            torch.abs(dof_vel[:, self.right_hip_joint_idx[0]]))
        # rew_dof_vel = rew_scale_dof_vel * hip_vel.squeeze(-1)

        # 左右脚离地奖励
        rew_left_airtime = rew_scale_airtime * self.left_foot_airtime
        rew_right_airtime = rew_scale_airtime * self.right_foot_airtime
        rew_airtime = rew_left_airtime + rew_right_airtime

        # Compute total reward
        total_reward = (rew_alive + 
                        rew_termination + 
                        rew_dist + 
                        rew_direction + 
                        rew_vel + 
                        # rew_dof_vel +
                        rew_airtime
                        )
        self.extras["log"] = {
        "rew_alive": (rew_alive).mean(),
        "rew_termination": (rew_termination).mean(),
        "rew_dist": (rew_dist).mean(),
        "rew_direction": (rew_direction).mean(),
        "rew_vel": (rew_vel).mean(),
        # "rew_dof_vel": (rew_dof_vel).mean(),
        "rew_airtime": (rew_airtime).mean(),
        }
        return total_reward



def quaternion_to_angle(quaternions: torch.Tensor):
    if quaternions.shape[1] != 4:
        raise ValueError("Input quaternion must have shape (n, 4).")
    
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    
    x_prime = 1 - 2 * (y**2 + z**2)
    
    angles = torch.arccos(torch.clamp(x_prime, -1.0, 1.0))  # [-1, 1]
    # print("angle: ", angles, "shape:", angles.shape)

    return angles
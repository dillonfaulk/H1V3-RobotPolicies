# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a trained v1x1 omni-directional checkpoint, driven LIVE by a ROS2 /cmd_vel topic instead of the
training task's randomly-sampled velocity commands.

AXIS MAPPING (v1x1-specific, see project memory for full derivation): v1x1's real anatomical forward is
local +Y, not the +X most Isaac Lab robots use. So:
    ROS Twist.linear.x  (forward/back) -> command lin_vel_y slot
    ROS Twist.linear.y  (left/right strafe) -> command lin_vel_x slot
    ROS Twist.angular.z (yaw rate) -> command ang_vel_z slot (unchanged, same convention both sides)

Drive it with, in a separate terminal (after sourcing ROS2 Jazzy):
    ros2 run teleop_twist_keyboard teleop_twist_keyboard

Usage:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_v1x1_cmd_vel.py \
        --checkpoint /path/to/v1x1_omni_final.pt --num_envs 1
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a v1x1 checkpoint driven by ROS2 /cmd_vel.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pt file.")
parser.add_argument(
    "--cmd_vel_topic", type=str, default="/cmd_vel", help="ROS2 topic to subscribe to for velocity commands."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# everything below needs the simulation app already running
import threading

import torch

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
import importlib.metadata as metadata

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.manager_based.classic.v1x1.agents.rsl_rl_ppo_cfg import V1X1VelocityPPORunnerCfg
from isaaclab_tasks.manager_based.classic.v1x1.v1x1_velocity_env_cfg import V1X1VelocityEnvCfg
from rsl_rl.runners import OnPolicyRunner

import gymnasium as gym


class CmdVelSubscriber(Node):
    """Thread-safe holder for the latest /cmd_vel message, mapped into v1x1's command axis convention."""

    def __init__(self, topic: str):
        super().__init__("v1x1_cmd_vel_bridge")
        self._lock = threading.Lock()
        # [lin_vel_x (lateral), lin_vel_y (v1x1 forward), ang_vel_z]
        self._command = [0.0, 0.0, 0.0]
        self.create_subscription(Twist, topic, self._callback, 10)
        self.get_logger().info(f"Subscribed to {topic}, mapping linear.x->v1x1 forward (lin_vel_y).")

    def _callback(self, msg: Twist):
        with self._lock:
            self._command[0] = msg.linear.y  # strafe -> v1x1 lateral
            self._command[1] = msg.linear.x  # forward -> v1x1's real forward axis
            self._command[2] = msg.angular.z

    def get_command(self) -> list[float]:
        with self._lock:
            return list(self._command)


def main():
    rclpy.init()
    node = CmdVelSubscriber(args_cli.cmd_vel_topic)
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    env_cfg: ManagerBasedRLEnvCfg = V1X1VelocityEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # standing envs / resampling don't matter here -- we override the command every step regardless.
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0

    env = gym.make("Isaac-V1X1-Velocity-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=100.0)

    resume_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    agent_cfg = V1X1VelocityPPORunnerCfg()
    agent_cfg.device = env.unwrapped.device
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    command_term = env.unwrapped.command_manager.get_term("base_velocity")

    obs = env.get_observations()
    print("[INFO]: Ready. Drive with `ros2 run teleop_twist_keyboard teleop_twist_keyboard` in another terminal.")
    try:
        while simulation_app.is_running():
            live_cmd = node.get_command()
            cmd_tensor = torch.tensor(live_cmd, device=env.unwrapped.device).repeat(args_cli.num_envs, 1)
            command_term.vel_command_b[:] = cmd_tensor
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                policy.reset(dones)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

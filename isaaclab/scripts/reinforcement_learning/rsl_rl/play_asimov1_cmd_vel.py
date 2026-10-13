# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a trained Isaac-Asimov1-Velocity-AMP-v0 checkpoint, driven LIVE by a ROS2 /cmd_vel topic instead of
the training task's randomly-sampled velocity commands. Mirrors v1x1's own play_v1x1_cmd_vel.py bridge.

AXIS MAPPING (asimov-1-specific, confirmed in asimov1_velocity_env_cfg.py's own "FORWARD-AXIS CHECK" --
derived from the URDF's own foot collision geometry, not assumed): asimov-1's real anatomical forward is
local +X, matching Isaac Lab's own H1/G1 convention -- unlike v1x1, which needed +X/+Y swapped. So this
mapping is a direct, unmodified passthrough:
    ROS Twist.linear.x  (forward/back)      -> command lin_vel_x slot
    ROS Twist.linear.y  (left/right strafe) -> command lin_vel_y slot
    ROS Twist.angular.z (yaw rate)          -> command ang_vel_z slot

Mirrors train_asimov1_velocity_amp_direct.py's / play_asimov1_velocity_amp.py's AppLauncher-first
construction order -- same stock-CLI import-order crash risk applies to any brand-new gym id.

Drive it with, in a separate terminal (after sourcing ROS2 Jazzy):
    ros2 run teleop_twist_keyboard teleop_twist_keyboard

Usage:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_asimov1_cmd_vel.py \
        --checkpoint /path/to/model_XXXX.pt --num_envs 1
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained asimov-1 checkpoint driven by ROS2 /cmd_vel.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pt file.")
parser.add_argument(
    "--cmd_vel_topic", type=str, default="/cmd_vel", help="ROS2 topic to subscribe to for velocity commands."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# everything below needs the simulation app already running -- this ordering is the actual fix.
import threading

import torch

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.classic.asimov1.agents.rsl_rl_ppo_cfg import ASIMOV1VelocityAmpPPORunnerCfg
from isaaclab_tasks.manager_based.classic.asimov1.asimov1_velocity_amp_env_cfg import ASIMOV1VelocityAmpEnvCfg

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner
import importlib.metadata as metadata

import gymnasium as gym


class CmdVelSubscriber(Node):
    """Thread-safe holder for the latest /cmd_vel message, direct passthrough into asimov-1's command axes
    (no remap needed -- asimov-1's real forward is local +X, same convention ROS Twist.linear.x already
    uses)."""

    def __init__(self, topic: str):
        super().__init__("asimov1_cmd_vel_bridge")
        self._lock = threading.Lock()
        # [lin_vel_x (forward), lin_vel_y (strafe), ang_vel_z]
        self._command = [0.0, 0.0, 0.0]
        self.create_subscription(Twist, topic, self._callback, 10)
        self.get_logger().info(f"Subscribed to {topic}, direct passthrough (linear.x = asimov-1 forward).")

    def _callback(self, msg: Twist):
        with self._lock:
            self._command[0] = msg.linear.x
            self._command[1] = msg.linear.y
            self._command[2] = msg.angular.z

    def get_command(self) -> list[float]:
        with self._lock:
            return list(self._command)


def main():
    rclpy.init()
    node = CmdVelSubscriber(args_cli.cmd_vel_topic)
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    cfg = ASIMOV1VelocityAmpEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    # standing envs / resampling don't matter here -- we override the command every step regardless.
    cfg.commands.base_velocity.rel_standing_envs = 0.0

    env = gym.make("Isaac-Asimov1-Velocity-AMP-v0", cfg=cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=100.0)

    resume_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    agent_cfg = ASIMOV1VelocityAmpPPORunnerCfg()
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

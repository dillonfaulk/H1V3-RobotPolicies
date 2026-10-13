# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play back a trained Isaac-Asimov1-Velocity-AMP-v0 checkpoint live, driven by the task's own randomly
sampled velocity commands (no override) -- for visually reviewing a checkpoint before deciding whether to
keep training on top of it.

Mirrors train_asimov1_velocity_amp_direct.py's AppLauncher-first construction order (see that file's
docstring for why the stock CLI/play scripts can't be trusted for this gym id -- same import-order risk:
the stock play_rsl_rl.py also imports isaaclab_tasks at module level before AppLauncher is constructed).

Usage:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_asimov1_velocity_amp.py \
        --checkpoint /path/to/model_XXXX.pt --num_envs 16
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained Isaac-Asimov1-Velocity-AMP-v0 checkpoint live.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pt file.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# everything below needs the simulation app already running -- this ordering is the actual fix.
import torch

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.classic.asimov1.agents.rsl_rl_ppo_cfg import ASIMOV1VelocityAmpPPORunnerCfg
from isaaclab_tasks.manager_based.classic.asimov1.asimov1_velocity_amp_env_cfg import ASIMOV1VelocityAmpEnvCfg

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner
import importlib.metadata as metadata


def main():
    cfg = ASIMOV1VelocityAmpEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs

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

    obs = env.get_observations()
    print("[INFO]: Ready, playing back live with the task's own sampled velocity commands.")
    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                policy.reset(dones)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

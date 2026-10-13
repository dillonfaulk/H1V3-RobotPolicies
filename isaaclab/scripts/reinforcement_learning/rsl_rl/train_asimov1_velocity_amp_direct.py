# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone trainer for Isaac-Asimov1-Velocity-AMP-v0, mirroring train_asimov1_velocity_direct.py's
AppLauncher-first construction order (see train_asimov1_direct.py's docstring for why the stock CLI
can't be used for a brand-new asimov1 gym id).

FRESH task, not a resume of either the standing or plain-velocity lineages -- different observation space
(adds an `amp` group) AND a different algorithm class (AMPPPO instead of plain PPO, with its own
discriminator/replay-buffer state that a plain-PPO checkpoint has no slot for).

Usage:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_asimov1_velocity_amp_direct.py \
        --num_envs 128 --max_iterations 300 --viz kit \
        --kit_args="--/renderer/multiGpu/enabled=False --/renderer/activeGpu=0"
"""

import argparse
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Standalone asimov1 AMP velocity-task trainer.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to simulate.")
parser.add_argument(
    "--max_iterations",
    type=int,
    default=300,
    help="Number of training iterations to run. ADDITIVE on top of a --checkpoint's saved iteration count.",
)
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument(
    "--checkpoint", type=str, default=None, help="Path to an Isaac-Asimov1-Velocity-AMP-v0 .pt checkpoint to resume from (optional)."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# everything below needs the simulation app already running -- this ordering is the actual fix.
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
    cfg.seed = args_cli.seed

    env = gym.make("Isaac-Asimov1-Velocity-AMP-v0", cfg=cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=100.0)

    agent_cfg = ASIMOV1VelocityAmpPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.device = env.unwrapped.device
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    log_dir = f"logs/rsl_rl/{agent_cfg.experiment_name}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        print(f"[INFO]: Resuming from checkpoint: {resume_path}")
        runner.load(resume_path)

    print(f"[INFO]: Logging experiment to directory: {log_dir}")
    runner.learn(num_learning_iterations=args_cli.max_iterations)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

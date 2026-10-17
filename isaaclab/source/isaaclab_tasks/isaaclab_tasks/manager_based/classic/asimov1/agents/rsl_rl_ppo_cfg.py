# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from ..amp.motion_dataset import MotionDatasetCfg
from ..asimov1_velocity_amp_env_cfg import (
    ASIMOV1_ANCHOR_NAME,
    ASIMOV1_KEY_BODY_NAMES,
    ASIMOV1_MOTION_FILES,
    ASIMOV1_AMP_OBS_TERMS,
    ASIMOV1_JOINT_NAMES,
)


@configclass
class ASIMOV1PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 1000
    save_interval = 50
    experiment_name = "asimov1"
    actor = RslRlMLPModelCfg(
        hidden_dims=[400, 200, 100],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[400, 200, 100],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class AMPPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Extends the stock PPO algorithm cfg with AMP's extra hyperparameters. class_name points
    OnPolicyRunner's own `resolve_callable(cfg["algorithm"]["class_name"])` plug-in mechanism at our ported
    amp/amp_ppo.py:AMPPPO -- confirmed this mechanism exists in our installed rsl-rl-lib 5.0.1 by reading
    OnPolicyRunner.__init__ directly before porting anything (see asimov1_velocity_amp_env_cfg.py's module
    docstring for the full story)."""

    class_name: str = "isaaclab_tasks.manager_based.classic.asimov1.amp.amp_ppo:AMPPPO"

    amp_data: MotionDatasetCfg = MISSING
    amp_obs_key: str = "amp"
    # Same starting values cyclotron itself ships with -- these are THEIR validated defaults for this exact
    # robot's AMP setup, not a guess, so start here rather than re-deriving from scratch.
    amp_reward_coef: float = 0.3
    amp_task_reward_lerp: float = 0.7
    amp_discr_hidden_dims: list = [256, 256]
    amp_discr_activation: str = "relu"
    amp_feature_normalization: bool = True
    amp_grad_pen_lambda: float = 10.0
    amp_replay_buffer_size: int = 100_000
    amp_discr_trunk_weight_decay: float = 1.0e-3
    amp_discr_head_weight_decay: float = 1.0e-1
    amp_update_interval: int = 1
    # Gated on the velocity command's magnitude (not ungated like the plain default) -- our task already has
    # a real "base_velocity" command (cyclotron's own gate looks for "twist" first, then falls back to
    # "base_velocity", confirmed by reading amp_ppo.py's _get_velocity_command), so the style reward only
    # applies when actually commanded to move, not during the rel_standing_envs fraction.
    amp_reward_command_gate: bool = True
    amp_reward_command_name: str | None = "base_velocity"
    amp_reward_command_threshold: float = 0.1
    amp_rollout_obs_clip: float | None = 500.0
    amp_min_normalized_std: float | None = 0.0


@configclass
class ASIMOV1VelocityAmpPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """AMP variant of the velocity-command runner -- separate experiment_name so checkpoints land in their
    own logs/rsl_rl/asimov1_velocity_amp/ directory (different observation space AND different algorithm
    class, cannot be mixed with either the plain-velocity or standing lineages)."""

    num_steps_per_env = 32
    max_iterations = 1000
    save_interval = 50
    experiment_name = "asimov1_velocity_amp"
    actor = RslRlMLPModelCfg(
        hidden_dims=[400, 200, 100],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[400, 200, 100],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = AMPPPOAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # Raised 0.0 -> 0.01 live mid-session: action_std had been in steady free-fall across every AMP run
        # (0.80 -> 0.63 -> 0.59 -> 0.53 -> 0.51 -> 0.42 -> 0.37) with nothing counteracting it, since
        # entropy_coef=0.0 means no explicit reward for maintaining exploration -- pure PPO convergence,
        # unopposed. Removing the hip_torso_rotation conflict (previous fix) didn't move error_vel_xy at
        # all in the very next run, while error_vel_yaw (a smaller behavioral change) kept improving freely
        # -- consistent with the policy being too converged/narrow to rediscover a genuinely different
        # strategy (real forward translation) even once the structural obstacle was removed. A small
        # entropy bonus directly counteracts the std collapse, giving the policy room to actually explore
        # committed strides rather than settling into "balance well, barely move."
        #
        # DIALED BACK 0.01 -> 0.003 several runs later: it did its job and then overshot. action_std kept
        # climbing every single run after this was introduced (0.52 -> 0.60 -> 0.76 -> 0.75 -> 0.93 -> 1.05)
        # -- past a point, MORE action noise doesn't help exploration, it actively prevents the policy from
        # executing and repeating a coherent plan. By the last run, action_std (1.05) exceeded even the
        # policy's own INITIAL value (init_std=1.0) before any training happened -- the policy had become
        # MORE random than when it started. This plausibly explains persistent "wiggling, not walking":
        # real stepping may be getting rediscovered moment-to-moment but knocked back into randomness
        # before it can be repeated/refined. discriminator_prediction also kept getting worse across the
        # same runs (-0.60 -> -0.84), consistent with increasingly noisy, less coherent motion. Lowering
        # (not removing, to avoid re-triggering the original collapse) to let it start consolidating.
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        amp_data=MotionDatasetCfg(
            motion_files=list(ASIMOV1_MOTION_FILES),
            joint_names=list(ASIMOV1_JOINT_NAMES),
            body_names=list(ASIMOV1_KEY_BODY_NAMES),
            amp_obs_terms=list(ASIMOV1_AMP_OBS_TERMS),
            anchor_name=ASIMOV1_ANCHOR_NAME,
        ),
    )


@configclass
class ASIMOV1FallRecoveryPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """First asimov1 runner cfg to set `obs_groups` explicitly -- the asymmetric actor/critic wiring this
    task needs (critic gets the "privileged" group's extra contact-wrench/CoM-velocity terms on top of the
    actor's own proprioception-only "policy" group). Separate experiment_name -- checkpoints land in their
    own logs/rsl_rl/asimov1_fall_recovery/ directory.

    `entropy_coef`/`obs_normalization`/`learning_rate`/`max_iterations` all changed for run 14 after
    directly comparing against OpenHorizon's own getup-specific PPO config (openhorizon-labs/asimov-v1-
    policies, `src/asimov_policies/tasks/getup/rl_cfg.py`, confirmed via a direct fetch -- not inferred
    from their locomotion policy's own separate config). Their getup policy reaches 100% sim success;
    13 straight runs here kept hitting the same sitting local optimum despite reward-shaping changes that
    tracked their design closely (same gating patterns, same assist-force concept) -- the gap looks less
    like a missing reward term and more like three concrete, confirmed hyperparameter/scale differences:
    1. **Scale**: their run is `max_iterations=20_000` * `num_envs=4096` * `num_steps_per_env=24` ~= 1.97
       BILLION total env-steps. Ours (1500*256*32 ~= 12.3M) was ~160x less total training experience --
       almost certainly the dominant factor. `max_iterations` raised 1500 -> 4000 here (num_envs kept at
       256, not scaled up, to stay within this project's own proven live-viewing render/VRAM budget) --
       still far short of their real budget, but a meaningful step up within a realistic single local
       session (~1.6-2 hours at this run's own measured iteration rate).
    2. **`entropy_coef`**: theirs is 0.005 for getup specifically; ours was 0.0. This directly controls how
       much exploration pressure persists through training -- a nonzero entropy bonus keeps pushing against
       exactly the kind of "found something comfortable (sitting), stop trying" premature convergence this
       task keeps hitting. Raised 0.0 -> 0.005 to match.
    3. **`obs_normalization`**: theirs is True for getup specifically (their own comment: "Differs from
       locomotion: normalization is baked into the ONNX export") -- every existing checkpoint in this WHOLE
       project (asimov1 omni/v1x1/BDX-R) was trained with this False, confirmed fact from the H1V3-
       RobotPolicies conversion work. This is the FIRST task to flip it True, matching their own explicit
       choice for this exact task (not locomotion). `learning_rate` also raised 5e-4 -> 1e-3 to match.
    """

    num_steps_per_env = 32
    max_iterations = 4000
    save_interval = 50
    experiment_name = "asimov1_fall_recovery"
    obs_groups = {"actor": ["policy"], "critic": ["policy", "privileged"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[400, 200, 100],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[400, 200, 100],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class ASIMOV1VelocityPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Separate experiment_name from the standing task -- checkpoints land in their own
    logs/rsl_rl/asimov1_velocity/ directory, never mixing with the stand-and-balance lineage (same
    fresh-start/separate-logging precedent v1x1's own velocity pivot used)."""

    num_steps_per_env = 32
    max_iterations = 1000
    save_interval = 50
    experiment_name = "asimov1_velocity"
    actor = RslRlMLPModelCfg(
        hidden_dims=[400, 200, 100],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[400, 200, 100],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

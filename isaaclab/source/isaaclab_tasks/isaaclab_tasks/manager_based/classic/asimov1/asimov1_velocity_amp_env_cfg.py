# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2026, Menlo Research Pte. Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMP (Adversarial Motion Priors) variant of the velocity-command task.

CONTEXT: while tuning Isaac-Asimov1-Velocity-v0's hand-shaped reward live, the user asked whether Menlo
Research (asimov-1's creator) had published their own training data. A web search found
`menloresearch/cyclotron`, their official Isaac Lab training repo, which DOES include a real reference
motion clip (`amp/motions/policy_delay_walk_slow.npz` -- ported into this project's own `amp/` subpackage,
not just linked) and a full AMP algorithm built on top of it: a discriminator that learns to tell "real
walking frame" from "policy-generated frame," with the policy rewarded for fooling it. Menlo's own blog post
on this (https://menlo.ai/blog/teaching-a-humanoid-to-walk) explains WHY: they deliberately removed a
sine/cosine gait-clock phase signal from their reward shape and let AMP's own discriminator define "looks
like walking" instead of hand-specifying a phase/rhythm -- directly addressing this project's own earlier
finding that our `feet_air_time_positive_biped` reward is looser than v1x1's phase-locked TrotGaitReward
(see asimov1_velocity_env_cfg.py's module docstring).

COMPATIBILITY CHECK (done before porting any code, not assumed): `cyclotron`'s AMPPPO is a plain
`rsl_rl.algorithms.PPO` subclass that plugs in through `OnPolicyRunner`'s own `cfg["algorithm"]["class_name"]`
resolution mechanism -- confirmed by reading our installed rsl-rl-lib 5.0.1's `OnPolicyRunner.__init__`
directly, which already has this exact plug-in point built in. No rsl_rl modifications were needed, just
pointing `AMPPPOAlgorithmCfg.class_name` at our ported `amp/amp_ppo.py:AMPPPO`.

NOTABLE DISCOVERY WORTH FLAGGING (separate from the AMP port itself): Menlo's own default standing pose
(`ASIMOV_1_STANDING_INIT_STATE` in their `assets/robots/asimov_1.py`) is a bent-knee "athletic" stance
(hip_pitch +-0.15, knee +-0.45, ankle_pitch +-0.30, pos.z=0.639) -- NOT the all-zero straight-leg pose this
project's own ASIMOV1_CFG uses. Straight-knee stances are a known hard case for bipedal control (singular/
near-singular leg Jacobian at full extension), and real sim-to-real humanoid work almost always uses a
slight crouch for exactly this reason (H1/G1's own default poses aren't fully straight either). Adopted HERE
(this AMP task needs a fresh start anyway, so there's no extra cost to also fixing this), but NOT
retroactively applied to Isaac-Asimov1-Stand-v0 or the plain Isaac-Asimov1-Velocity-v0 -- that's a separate
decision with its own fresh-start cost, flagged for the user rather than applied silently.

Everything else (scene, actions, most rewards/terminations, the feet-separation/stance-width/recovery-step
family) is inherited unchanged from Isaac-Asimov1-Velocity-v0 -- this file only ADDS the `amp` observation
group that feeds the discriminator, and swaps in the bent-knee init pose above.
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from .asimov1_velocity_env_cfg import ASIMOV1VelocityEnvCfg
from .asimov1_velocity_env_cfg import ObservationsCfg as _BaseObservationsCfg

# Bent-knee "athletic" default pose -- see module docstring. Matches Menlo's own ASIMOV_1_STANDING_INIT_STATE
# exactly (their assets/robots/asimov_1.py), not re-derived -- their team validated it, no reason to guess
# our own when a real answer already exists.
# Leg bend angles scaled down ~30% from Menlo's own values (0.7x) per earlier user feedback ("straighten
# the legs just a tad") -- then nudged back UP to ~0.85x live mid-session per direct follow-up feedback
# during this second fresh restart ("knees are a little too straight. that might be hindering them") --
# 0.7x turned out to be an overcorrection. hip_pitch/knee/ankle_pitch still scaled TOGETHER (not just the
# knee alone) to keep the stance geometrically consistent. Still short of Menlo's full original depth --
# the bend itself is intentional (avoids the singular-Jacobian problem a fully straight leg has for
# control), just shallower than their exact value. height left unchanged at 0.639, still flagged pending
# visual confirmation like every other numeric placeholder in this project.
ASIMOV1_AMP_INIT_STATE_POS = (0.0, 0.0, 0.639)
ASIMOV1_AMP_INIT_STATE_JOINT_POS = {
    "left_hip_pitch_joint": -0.13,
    "right_hip_pitch_joint": 0.13,
    ".*_hip_roll_joint": 0.0,
    ".*_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.38,
    "right_knee_joint": -0.38,
    "left_ankle_pitch_joint": -0.26,
    "right_ankle_pitch_joint": 0.26,
    ".*_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "left_shoulder_pitch_joint": -0.25,
    "right_shoulder_pitch_joint": 0.25,
    "left_shoulder_roll_joint": -0.05,
    "right_shoulder_roll_joint": 0.05,
    ".*_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.40,
    "right_elbow_joint": -0.40,
    ".*_wrist_yaw_joint": 0.0,
}

# Shared between the AMP observation group's SceneEntityCfg and MotionDatasetCfg (agents/rsl_rl_ppo_cfg.py)
# -- motion_dataset.py reindexes the .npz's own columns by NAME lookup, so this list's order doesn't need to
# match the file's internal order, it only needs to be the SAME list used on both sides.
ASIMOV1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
]

# body_names/anchor_name are required by MotionDataset's __init__ for index resolution even though our
# amp_obs_terms only use joint_pos/joint_vel (not body-relative features) -- see amp/motion_dataset.py.
ASIMOV1_KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "waist_yaw_link",
]
ASIMOV1_ANCHOR_NAME = "pelvis_link"
ASIMOV1_AMP_OBS_TERMS = ["joint_pos", "joint_vel"]

_MOTIONS_DIR = Path(__file__).resolve().parent / "amp" / "motions"
ASIMOV1_MOTION_FILES = [str(_MOTIONS_DIR / "policy_delay_walk_slow.npz")]

_AMP_JOINTS_CFG = SceneEntityCfg("robot", joint_names=list(ASIMOV1_JOINT_NAMES), preserve_order=True)


@configclass
class AmpObsCfg(ObsGroup):
    """Feature group the discriminator compares against the reference motion. MUST stay exactly
    [joint_pos, joint_vel] in this order -- enforced below so the discriminator's policy-side and
    expert-side features never silently drift apart (mirrors cyclotron's own assertion in amp_env_cfg.py)."""

    joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _AMP_JOINTS_CFG})
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _AMP_JOINTS_CFG})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        term_names = [name for name, value in self.__dict__.items() if isinstance(value, ObsTerm)]
        assert term_names == ASIMOV1_AMP_OBS_TERMS, (
            f"AMP obs group terms {term_names} drifted from ASIMOV1_AMP_OBS_TERMS {ASIMOV1_AMP_OBS_TERMS}; "
            "the discriminator's policy and expert features would be misaligned."
        )


@configclass
class ObservationsCfg(_BaseObservationsCfg):
    """Same policy observations as the plain velocity task, plus the amp group for the discriminator."""

    amp: AmpObsCfg = AmpObsCfg()


@configclass
class Asimov1VelocityAmpObservationsCfg(PresetCfg):
    default: ObservationsCfg = ObservationsCfg()
    physx: ObservationsCfg = ObservationsCfg()
    newton_mjwarp: ObservationsCfg = ObservationsCfg()


@configclass
class ASIMOV1VelocityAmpEnvCfg(ASIMOV1VelocityEnvCfg):
    """Isaac-Asimov1-Velocity-v0 + an `amp` observation group feeding the AMP discriminator, + the bent-knee
    athletic default pose (see module docstring)."""

    observations: Asimov1VelocityAmpObservationsCfg = Asimov1VelocityAmpObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.init_state.pos = ASIMOV1_AMP_INIT_STATE_POS
        self.scene.robot.init_state.joint_pos = dict(ASIMOV1_AMP_INIT_STATE_JOINT_POS)

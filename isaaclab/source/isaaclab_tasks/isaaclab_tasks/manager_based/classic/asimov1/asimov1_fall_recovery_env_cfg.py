# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fall-recovery (getup) task for asimov-1 -- a brand new, separate task from the stand-and-balance and
walking lineages. Nothing in asimov1_env_cfg.py/asimov1_velocity_env_cfg.py/asimov1_velocity_amp_env_cfg.py
is touched; this file only imports the pieces that genuinely transfer (MySceneCfg, Asimov1ContactSensorCfg,
Asimov1PhysicsCfg -- same reuse precedent asimov1_velocity_env_cfg.py already established).

No stock Isaac Lab fall-recovery/getup task exists anywhere in this checkout (confirmed by direct search --
unlike the walking task, there was no H1/G1/Spot template to mirror here). No local reference getup motion
clip exists either (asimov1's AMP motion library only has `policy_delay_walk_slow.npz`), so this is plain
PPO, matching v1x1/BDX-R precedent -- not AMP.

Built from literature research the user did on 6 papers (HumanUP arXiv:2502.12152, HoST arXiv:2502.08378, a
Capture-Point/DCM asymmetric-critic paper arXiv:2603.08619, FRASA arXiv:2410.08655, FIRM arXiv:2511.07407,
Stubborn arXiv:2606.12814), deliberately scoped to the two SIMPLEST papers' approach (HumanUP, HoST: pure PPO
+ curriculum, no demonstrations/off-policy algorithms) for v1. Capture Point/DCM, CrossQ, and demonstration-
augmentation are explicitly OUT of scope -- possible future refinements once a working baseline exists.

KEY DESIGN DEPARTURES FROM EVERY OTHER ASIMOV1 TASK (read before touching):
- Terminations are `time_out` ONLY. The stand/walk tasks terminate on pelvis contact, collapsed height, bad
  orientation, feet crossed/too-wide -- every one of those would fire on EVERY step here by design, since
  starting on and interacting with the ground is the entire premise of this task.
- Action scale raised 0.25 -> 0.5 (vs. the stand task) -- a getup needs much bigger joint excursions than
  balance micro-corrections; HumanUP's own Phase-1 philosophy ("let the policy freely discover ANY path,
  even high-torque") argues for loosening this. User-confirmed decision, not silently assumed.
- No explicit settle-phase reward gating -- relies on every reward term already being bounded/graded rather
  than cliff-edged, and the brief 0.2-0.4m-equivalent free-fall being a tiny fraction of a 12s episode.
  User-confirmed decision; revisit only if a real run shows a problem.
- Asymmetric actor/critic: actor is pure proprioception (deployable), critic additionally gets per-body
  contact wrench (8 bodies: feet+hand-ends+knees+pelvis+torso) + base_lin_vel. This is the FIRST asimov1 task
  using `obs_groups` to wire this asymmetry (confirmed: none of the other three set it, single "policy"
  group only) -- verify it actually wires the critic's input width correctly during the smoke test, don't
  trust the config alone.
- Two-phase curriculum (HumanUP): Phase 1 = relaxed smoothness/torque/contact-force penalty weights (let the
  policy discover ANY path to standing). Phase 2 = `mdp.modify_reward_weight` (stock, already proven in
  `lift_env_cfg.py`/`reach_env_cfg.py`) ramps those weights up once `common_step_counter` crosses a
  threshold, ramping up smoothness once upright poses are reliably found.
"""

import isaaclab.envs.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loco_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import JointWrenchSensorCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from . import mdp as asimov1_mdp
from .asimov1_env_cfg import Asimov1PhysicsCfg, MySceneCfg as _BaseSceneCfg

# the 8 contact-relevant bodies for this task: feet + hand-ends (no gripper/hand asset exists yet -- wrist
# is the last arm link) + knees + pelvis + torso. Scoped to what asimov-1 physically has today.
FALL_RECOVERY_CONTACT_BODIES = [
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
    "left_knee_link", "right_knee_link",
    "pelvis_link",
    "waist_yaw_link",
]

# the `contact_forces` sensor (PhysX contact-report API) was only ever activated, at this asset's one-time
# USD conversion, on the 3 bodies the EXISTING stand/walk tasks needed (feet + pelvis) -- confirmed directly
# via a dry-run crash: "Available strings: ('pelvis_link', 'left_ankle_roll_link', 'right_ankle_roll_link')".
# Knees/wrists/waist never had contact-report enabled on this asset. Narrowed here for `contact_force_limit`
# specifically; `joint_wrench` (a different mechanism -- reads joint reaction forces directly, not gated by
# a per-body contact-report opt-in) is tried with the FULL 8-body set below and may not have this limitation.
CONTACT_SENSOR_AVAILABLE_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link", "pelvis_link"]

# asimov-1's own default standing-height estimate (see robots/asimov1.py init_state) -- same constant the
# stand/walk tasks' own base_height reward already uses.
STANDING_HEIGHT = 0.63

# The WALKING policy's own default/init joint pose (bent-knee "athletic" stance, Menlo's own
# ASIMOV_1_STANDING_INIT_STATE) -- NOT this task's own all-zero default. Copied verbatim from
# `WALKING_DEFAULT_JOINT_POS` in `scripts/.../asimov1_phase_stitch/observations.py`, which was itself
# confirmed by direct instantiation of Isaac-Asimov1-Velocity-AMP-v0 (`env.scene["robot"].data.
# default_joint_pos`), in the asset-intrinsic joint order (same order `asset.data.joint_pos` uses here --
# both tasks share the identical ASIMOV1_CFG asset, confirmed earlier this project via a programmatic
# joint-name comparison). Target for `recovery_pose_alignment_exp` below: added per explicit user insight
# that the recovery policy's final pose only ever had to look "standing" by this task's own generic
# height/orientation/stability criteria, with NO notion of matching what the WALKING policy actually
# expects its own starting joint configuration to look like -- exactly the kind of joint-space mismatch
# that stayed invisible to every root-level phase-stitch gate while still causing real handoff failures.
WALKING_HANDOFF_TARGET_JOINT_POS = [
    -0.13, 0.13, 0.0, 0.0, 0.0, 0.25, -0.25, 0.0, 0.0, 0.05, -0.05, 0.38, -0.38,
    0.0, 0.0, -0.26, 0.26, -0.4, 0.4, 0.0, 0.0, 0.0, 0.0,
]

# rough estimate for an upright-SITTING pelvis height (butt on the ground, torso vertical) -- a plausible
# easier-to-reach intermediate waypoint before standing, per the user's own insight: most real getup
# sequences involve sitting upright first (lower center of mass, bigger stable base) before attempting the
# harder kneel-or-stand transition. No real asset measurement behind this number (unlike STANDING_HEIGHT,
# which comes from the robot's own init_state) -- a rough proportional guess (roughly 25% of standing
# pelvis height, loosely matching typical human seated-vs-standing hip height ratios), flagged for visual
# verification once we can actually watch for it.
SIT_HEIGHT = 0.15

# rough estimate for a KNEELING pelvis height (knee(s) down, torso upright) -- a second intermediate
# waypoint between sitting and standing, added after run 9 showed robots reliably reaching SIT_HEIGHT with
# good orientation but then plateauing there for the back half of the run (height_tracking/sit_height_
# tracking both went flat, assist/scale never decayed once). Breaks the sit->stand transition (the hardest
# single jump in every cited getup paper) into two smaller, more learnable hops, matching the user's own
# earlier idea and the curriculum-staging note from the OpenHorizon getup-chain brief ("fall -> sit on butt
# -> get on knee/shin -> stand"). Same proportional-guess caveat as SIT_HEIGHT -- no real asset measurement
# behind this number, roughly 63% of standing height (typical human kneeling-vs-standing hip height ratio),
# flagged for visual verification once we can watch for it.
KNEEL_HEIGHT = 0.4

# hands (wrist-ends) + knees -- the limbs to reward actively pushing off the ground with, to counter the
# "A-pose" local optimum (robot resting on its head + legs as a stable tripod instead of climbing further,
# confirmed live via training video). Uses the `joint_wrench` sensor (not `contact_forces`, which doesn't
# have report-API coverage on these bodies -- see CONTACT_SENSOR_AVAILABLE_BODIES above).
LIMB_PUSHOFF_BODIES = ["left_wrist_yaw_link", "right_wrist_yaw_link", "left_knee_link", "right_knee_link"]

# hands only (subset of LIMB_PUSHOFF_BODIES above) -- for `hand_pushoff_force_bonus`'s own threshold-shaped
# reward, separate from the combined hand+knee `limb_pushoff_bonus` term. User's own insight after run 13:
# robots are currently either sitting or doing a belly-down worm -- neither involves a deliberate hands-
# push phase to help get the feet/knees under the body between those two states.
HAND_PUSHOFF_BODIES = ["left_wrist_yaw_link", "right_wrist_yaw_link"]

# the non-leg joints that should sit at their default (facing-forward, symmetric) value regardless of
# sit-vs-stand -- see `sit_torso_alignment_penalty`'s own docstring for why hip_pitch/knee/ankle_pitch are
# deliberately excluded (those three MUST deviate from default to actually sit).
SIT_TORSO_ALIGNMENT_JOINTS = [
    "waist_yaw_joint",
    ".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_ankle_roll_joint",
    ".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_yaw_joint",
]

# control-step duration (decimation * sim.dt, must match __post_init__ below) -- used to make the
# assist-force EventTerm fire every single control step (see apply_assist_force's own docstring).
CONTROL_DT = 2 / 120.0

# assist-force curriculum constants -- originally ported from OpenHorizon's own validated numbers
# (openhorizon-labs/asimov-v1-policies, GetUpParams: assist_weight_fraction=0.6, assist_initial_scale=1.0,
# assist_decay_step=0.05, assist_success_threshold=0.6).
# ASSIST_INITIAL_SCALE dropped 1.0 -> 0.0 per explicit user call after runs 14-17 (89% unassisted) plus
# this session's two further resumed retrains (both confirmed via their own logged `Curriculum/assist/scale`
# decaying to 0.0 within the first few hundred iterations of every resume, success_rate already 90%+ by
# then) -- the bootstrapping role this curriculum was built for is over; keeping it at 1.0 only means every
# future resumed run wastes its first few hundred iterations re-decaying a crutch the policy no longer
# needs. Setting the INITIAL value to 0.0 makes the whole assist_force EventTerm a permanent no-op from
# iteration 0 (apply_assist_force scales its lift force by this value) without deleting the mechanism's own
# code -- trivially reversible (bump back to 1.0) if a future from-scratch retrain ever needs it again.
ASSIST_WEIGHT_FRACTION = 0.6
ASSIST_INITIAL_SCALE = 0.0
ASSIST_DECAY_STEP = 0.05
ASSIST_SUCCESS_THRESHOLD = 0.6
# one episode's worth of control steps (12s / CONTROL_DT = 720) -- our only termination is time_out, so
# every env resets in sync every episode; this just needs to be <= that to catch each synchronized burst.
ASSIST_WINDOW_STEPS = 700


##
# Scene
##


@configclass
class FallRecoverySceneCfg(_BaseSceneCfg):
    """Adds the one sensor this task needs that no other asimov1 task does -- per-body contact wrench for
    the critic's privileged observations. `contact_forces` (inherited unchanged) already covers every body
    via its `.*` glob, so it needs no changes."""

    joint_wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Larger action scale than the stand task's 0.25 -- a getup needs much bigger joint excursions than
    balance micro-corrections (user-confirmed decision, see module docstring)."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Asymmetric actor/critic. The critic group only needs to contain the EXTRA terms beyond the actor's
    own -- `obs_groups = {"actor": ["policy"], "critic": ["policy", "privileged"]}` in the agent cfg takes
    the union, it doesn't need a duplicated copy of the actor's terms in here."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor -- pure proprioception, deployable on real hardware. No privileged info."""

        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_height = ObsTerm(func=mdp.base_pos_z)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Critic-only additions: 8-body contact wrench + base linear velocity (CoM-velocity proxy)."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        contact_wrenches = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.01,
            params={"sensor_cfg": SceneEntityCfg("joint_wrench", body_names=FALL_RECOVERY_CONTACT_BODIES)},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class Asimov1FallRecoveryObservationsCfg(PresetCfg):
    """Mirrors the PresetCfg-wrapping convention every other asimov1/v1x1 task's ObservationsCfg uses --
    content doesn't actually vary by backend, this is the established house pattern for this field."""

    default: ObservationsCfg = ObservationsCfg()
    physx: ObservationsCfg = ObservationsCfg()
    newton_mjwarp: ObservationsCfg = ObservationsCfg()


@configclass
class EventCfg:
    """NOT a subclass of the stand task's EventCfg -- every term differs (orientation sampling strategy,
    range, and push_robot is dropped entirely -- there's no standing balance to disturb here)."""

    # Full-SO(3) random orientation (confirmed: this function samples orientation uniformly regardless of
    # pose_range, via math_utils.random_orientation) + a height offset. pose_range["z"] is ADDITIVE on top
    # of the default root pos z (0.63, the robot's own standing height -- see robots/asimov1.py init_state),
    # so (-0.23, -0.03) targets an ABSOLUTE root spawn height of ~0.4-0.6m -- enough ground clearance for a
    # ~1.2m-tall robot in an arbitrary random orientation without being an excessively violent drop. Small
    # initial linear/angular velocity jitter on top, not a meaningful contributor compared to the orientation
    # randomization itself.
    reset_base = EventTerm(
        func=mdp.reset_root_state_with_random_orientation,
        mode="reset",
        params={
            "pose_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "z": (-0.23, -0.03)},
            "velocity_range": {
                "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1),
                "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2),
            },
        },
    )
    # exact default joint pose on reset, matching the stand task's own choice for the same reason: a clean,
    # repeatable starting pose while debugging the new reward shape. Joint-pose-at-reset diversity deferred
    # to a later pass (user-confirmed) -- orientation/height randomization is the primary diversity axis here.
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
    # no push_robot -- there is no standing balance to disturb in this task.

    # assist-force curriculum (see asimov1_mdp.apply_assist_force's own docstring) -- a continuous upward
    # lift on the pelvis, scaled by AssistCurriculum below and decayed as success rate rises.
    # interval_range_s=(CONTROL_DT, CONTROL_DT) + is_global_time=True makes this fire every control step
    # for every env (Isaac Lab's EventManager has no literal "every step" mode).
    # min_upright=0.3 (default) gates the lift off while badly tilted (e.g. the A-pose) -- added after run
    # 8's first launch (unconditional lift) made the A-pose WORSE by subsidizing pelvis-height reward
    # regardless of orientation. See apply_assist_force's own docstring for the live evidence.
    assist_force = EventTerm(
        func=asimov1_mdp.apply_assist_force,
        mode="interval",
        interval_range_s=(CONTROL_DT, CONTROL_DT),
        is_global_time=True,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["pelvis_link"]),
            "max_weight_fraction": ASSIST_WEIGHT_FRACTION,
        },
    )


@configclass
class RewardsCfg:
    """Phase-1 weights below are deliberately relaxed (small torque/action-rate/contact-force penalties) --
    CurriculumCfg ramps the regularization terms up once the policy has had room to discover a working
    getup motion, per HumanUP's two-phase approach. Deliberately omitted vs. the stand/walk tasks:
    `is_alive`/`termination_penalty` -- there's no early-ending exploit to close since `time_out` and
    `head_ground` (see TerminationsCfg) have no such exploit either -- see their own docstrings.

    This reward set has gone through several iterations as real runs surfaced an A-pose local optimum
    (robots resting head/shoulders+legs as a stable tripod that earns real partial height-tracking credit
    without ever climbing further) and two distinct reward-hacking exploits along the way -- see each
    term's own comment for its specific history. `orientation_upright` is the latest addition: after 5
    straight runs showed height_tracking/torso_upright both improving but stability_hold never breaking
    away from ~0, added a graded positive orientation reward (mirroring height_tracking's own exp-kernel
    shape) to give orientation the same sharp "getting close to the goal" gradient height already had --
    flat_orientation_l2's pure penalty shape, by contrast, barely distinguishes "very tilted" from
    "extremely tilted," both just keep growing."""

    # core upright signal -- stock term, reused unmodified from the stand/walk tasks.
    torso_upright = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)

    # graded positive counterpart to torso_upright above -- see RewardsCfg's own docstring for why this
    # was added. Weight is a first-pass placeholder, same std as height_tracking for consistency.
    orientation_upright = RewTerm(
        func=asimov1_mdp.orientation_upright_exp,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "std": 0.3},
    )

    # height tracking via exp kernel -- new function (mdp.py), no stock exp-kernel equivalent exists (only
    # base_height_l2, pure L2). A graded exp kernel fits better here than a harsh L2 penalty, since almost
    # every initial state starts far from the target height.
    height_tracking = RewTerm(
        func=asimov1_mdp.height_tracking_exp,
        weight=3.0,
        params={"target_height": STANDING_HEIGHT, "std": 0.3},
    )

    # second height-target aimed at SIT_HEIGHT instead of full standing -- an easier, nearer stepping-stone
    # sub-goal, per the user's own sitting-first insight (see SIT_HEIGHT's own comment).
    # Uses `height_milestone_exp`, NOT `height_tracking_exp` -- confirmed via a real verification run that
    # the two-sided version actively fights the climb once past the target: it saturated near its max
    # (1.92/2.0) while the REAL height_tracking term stayed flat (0.25), since a two-sided exp kernel's
    # reward mathematically DECREASES once height rises past an intermediate target, directly opposing the
    # whole point of this task. `height_milestone_exp` is one-sided (see its own docstring) -- ramps up
    # approaching SIT_HEIGHT from below, then stays at full credit for any height at or above it, so
    # continuing to climb toward standing never costs anything here.
    sit_height_tracking = RewTerm(
        func=asimov1_mdp.height_milestone_exp,
        weight=2.0,
        params={"target_height": SIT_HEIGHT, "std": 0.15},
    )

    # second waypoint between sitting and standing -- see KNEEL_HEIGHT's own comment. Reuses
    # `height_milestone_exp` directly, same one-sided (never-penalizes-climbing-past-it) function
    # sit_height_tracking uses -- no new mdp.py code needed, just a different target height.
    kneel_height_tracking = RewTerm(
        func=asimov1_mdp.height_milestone_exp,
        weight=2.0,
        params={"target_height": KNEEL_HEIGHT, "std": 0.15},
    )

    # sustained-stability bonus -- new stateful ManagerTermBase (mdp.py). Direct code answer to a known
    # getup-RL failure mode (HoST, arXiv:2502.08378): a policy popping upright for one frame then collapsing
    # earns ~0 here, only a real, HELD getup pays out.
    stability_hold = RewTerm(
        func=asimov1_mdp.StabilityHoldBonus,
        weight=5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_height": STANDING_HEIGHT,
            "height_tolerance": 0.08,
            "tilt_tolerance": 0.3,
            "required_hold_steps": 25,
        },
    )

    # discourage wild flailing -- reused as-is from the walking task, robot-agnostic.
    angular_momentum = RewTerm(func=asimov1_mdp.whole_body_angular_momentum_l2, weight=-0.03)

    # ADDED per explicit user observation (live-watched the trained policy "scoot across the ground with
    # their feet like sliding a little" while recovering) -- two separate, complementary mechanisms for
    # two separate aspects of the same visible phenomenon:
    # (a) per-step foot-DRAGGING while a foot is planted -- reused UNMODIFIED from the walking task, where
    # it's already proven (same body set, same mechanism, zero new code).
    # weight HALVED -0.25 -> -0.15 after the first real run: the measured handoff success rate REGRESSED
    # (80.0% -> 61.5% in the live phase-stitch test) despite the user's own visual confirmation that
    # sliding itself was visibly reduced -- i.e. this term (plus recovery_position_drift below) was likely
    # competing against the primary height/pose objectives hard enough to cost handoff quality, a real
    # multi-objective tradeoff, not noise (failure heights crept back toward the pre-retrain pattern).
    # Kept active (not removed) since the underlying behavior-shaping direction is confirmed correct by
    # direct observation -- just needs less relative pull, paired with the new recovery_pose_alignment_exp
    # term below which targets the ACTUAL handoff-relevant signal (final joint config) more directly.
    feet_slide = RewTerm(
        func=loco_mdp.feet_slide,
        weight=-0.15,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
        },
    )
    # (b) NET horizontal travel away from where the robot fell, over the whole episode -- a real
    # functional requirement per the user's own framing (household chores in tight hallways can't tolerate
    # "recovers, but ends up a meter from where it fell"). New stateful term, see
    # RecoveryPositionDriftPenalty's own docstring in mdp.py for the anchor mechanism and tolerance
    # reasoning. weight CUT -1.0 -> -0.4 for the same reason as feet_slide's own cut above (measured
    # handoff regression on the first real run) -- same "keep the direction, reduce the pull" fix.
    recovery_position_drift = RewTerm(
        func=asimov1_mdp.RecoveryPositionDriftPenalty,
        weight=-0.4,
        params={"asset_cfg": SceneEntityCfg("robot"), "tolerance": 0.15},
    )
    # NEW -- targets the ACTUAL handoff-relevant signal directly, added alongside the weight cuts above per
    # the user's own sharp insight: good recoveries vs. bad ones in the phase-stitch demo weren't
    # distinguishable by height/upright/ang_speed (confirmed via direct gate-value comparison -- every
    # failure looked statistically identical to a success on every root-level signal), so the real
    # remaining gap has to be in the exact final JOINT configuration, not anything visible at the root
    # level. See recovery_pose_alignment_exp's own docstring (mdp.py) for the gating/std reasoning.
    # weight=2.0 matches orientation_upright's own weight -- meant as a real primary shaping signal now,
    # not a minor nudge, since it directly targets the hypothesized root cause rather than a side effect.
    recovery_pose_alignment = RewTerm(
        func=asimov1_mdp.recovery_pose_alignment_exp,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_joint_pos": WALKING_HANDOFF_TARGET_JOINT_POS,
            "std": 1.5,
            "gate_height": 0.9 * STANDING_HEIGHT,
        },
    )

    # wide-body contact-force ceiling -- reused UNMODIFIED (confirmed not feet-specific despite the name, it
    # only ever reads sensor_cfg.body_ids generically). Phase-1-relaxed weight, ramped in CurriculumCfg.
    contact_force_limit = RewTerm(
        func=asimov1_mdp.feet_contact_force_limit,
        weight=-5.0e-6,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=CONTACT_SENSOR_AVAILABLE_BODIES),
            "max_force": 350.0,
        },
    )

    # direct CONTACT-PATTERN signal for standing (both feet down, pelvis not) -- see
    # feet_only_contact_bonus's own docstring. User's own idea after run 14's screenshot showed the whole
    # population converged on sitting despite excellent height/orientation numbers -- those can't
    # distinguish "sitting upright" from "standing," an actual contact pattern can. Weight is a first-pass
    # placeholder, watch the verification check's logged magnitude before trusting it on a live run.
    # contact_threshold raised 1.0 -> 20.0 AND height-gated after two verification checks showed the raw
    # contact pattern alone fires on ~70-83% of steps for a largely RANDOM policy (plenty of ordinary floor
    # postures satisfy "feet touching, pelvis not" without being anywhere near standing) -- see
    # feet_only_contact_bonus's own docstring for the full story and the OpenHorizon precedent this matches.
    feet_only_contact = RewTerm(
        func=asimov1_mdp.feet_only_contact_bonus,
        weight=1.5,
        params={
            "feet_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            ),
            "pelvis_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["pelvis_link"]),
            "standing_height": STANDING_HEIGHT,
            "transition_width": 0.15,
            "contact_threshold": 20.0,
        },
    )

    # smoothness / torque regularization -- stock terms, Phase-1-relaxed weights, ramped in CurriculumCfg.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    dof_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-7)

    # discourage using the head/shoulder region as a free ground-prop point -- see function docstring for
    # why `waist_yaw_link` + both shoulder links together are the best available proxy (no separate head
    # rigid body exists on this asset).
    # REPLACED the earlier `torso_near_ground_penalty` version (fixed min_height=0.25 threshold, zero
    # penalty once above it) -- a real run showed the logged penalty shrinking toward zero over training
    # while the user kept visually observing persistent head-resting live: the robot had found a resting
    # height just above the cutoff without the actual behavior stopping. This version is threshold-free and
    # fully graded (see upper_body_ground_penalty's docstring) -- no height at which it bottoms out to zero
    # before actually reaching standing height.
    # Weight raised back to -4.0 after a `head_ground` hard termination (tried as an alternative way to get
    # real teeth against this) failed twice in a row -- at grace_period=2.0/min_height=0.3 AND the much
    # looser grace_period=6.0/min_height=0.15, episode length stayed pinned at EXACTLY the grace-period
    # boundary for the entire 300-iteration verification run both times, zero escapes. Dropped that
    # termination entirely (same resolution as StalledRecoveryTermination -- see TerminationsCfg) and
    # putting the "stop this behavior harder" burden back on this already-proven-safe continuous penalty
    # instead. -4.0 was already smoke-tested once (making this term the single most dominant reward
    # component, -2.23, more than double the next-largest) and dialed back to -2.0 at the time for better
    # balance -- but now that it's the ONLY mechanism targeting this behavior (no termination backup),
    # that dominance is probably the right amount of signal, not an overcorrection. A dominant PENALTY
    # doesn't carry the same farming risk a dominant REWARD does (confirmed exploitable twice this session)
    # -- there's no way to "cash in" on successfully avoiding a penalty the way there is on a reward.
    upper_body_ground = RewTerm(
        func=asimov1_mdp.upper_body_ground_penalty,
        weight=-4.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["waist_yaw_link", "left_shoulder_pitch_link", "right_shoulder_pitch_link"]
            ),
            "target_height": STANDING_HEIGHT,
        },
    )

    # reward actually pushing off the ground with hands/knees WHILE rising -- gated on positive CoM
    # vertical velocity so resting weight on a limb while static earns nothing, only effort that coincides
    # with climbing does.
    # weight lowered 0.02 -> 0.004 after a real run showed this term reward-hacked: by iteration 250/1500
    # it had grown to be the single LARGEST reward component (weighted contribution 3.10, bigger than
    # height_tracking's 0.95) while angular_momentum's penalty simultaneously spiked (-0.86, vs. a typical
    # -0.05 to -0.08 elsewhere) and stability_hold stayed flat -- the signature of vigorous, uncoordinated
    # thrashing ("doing the worm," confirmed live) farming coincidental force+rising-velocity moments rather
    # than genuine controlled push-off. Combined with the vertical-component-only fix in mdp.py (same
    # finding), this reduced both the raw magnitude and how easily non-vertical thrashing could trigger it
    # -- confirmed via a focused 300-iteration headless check past the same iteration the exploit first
    # appeared, no recurrence. Raised slightly 0.004 -> 0.01 now that the fix is confirmed stable, per
    # explicit request to push this incentive a bit harder -- still well below the original exploited 0.02.
    limb_pushoff = RewTerm(
        func=asimov1_mdp.limb_pushoff_bonus,
        weight=0.01,
        params={"sensor_cfg": SceneEntityCfg("joint_wrench", body_names=LIMB_PUSHOFF_BODIES)},
    )

    # threshold-shaped bonus specifically for a DECISIVE hands push-off -- see hand_pushoff_force_bonus's
    # own docstring. Weight + min_force/max_force are all first-pass placeholders, watch the verification
    # check's logged magnitude before trusting them on a live run.
    # min_force/max_force lowered 30/150 -> 5/50 after run 14 showed this logged EXACTLY 0.0000 for all
    # 4000 iterations -- real evidence the original threshold was simply never crossed in combination with
    # positive rising velocity, not a sign the mechanism itself is broken (limb_pushoff/knee_bend_phase were
    # both actively engaging in that same run, confirming real push-off forces ARE happening). A much lower
    # band gives this term a chance to ever contribute at all; still gated the same safe way (vertical-
    # component-only, positive rising velocity) so this doesn't reopen the run-3 thrashing exploit.
    hand_pushoff_force = RewTerm(
        func=asimov1_mdp.hand_pushoff_force_bonus,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("joint_wrench", body_names=HAND_PUSHOFF_BODIES),
            "min_force": 5.0,
            "max_force": 50.0,
        },
    )

    # reward bending the knees while still low/recovering, PENALIZE bent knees once nearly standing -- see
    # function docstring for the phase-gate mechanism. User's own insight: knee flexion is the natural
    # lever for pushing up, but the same pose becomes wasted effort once actually upright. Weight is a
    # first-pass placeholder.
    knee_bend_phase = RewTerm(
        func=asimov1_mdp.knee_bend_phase_bonus,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_joint"]),
            "standing_height": STANDING_HEIGHT,
            "transition_width": 0.15,
        },
    )

    # penalizes the FULL joint array's deviation from default_joint_pos (asimov-1's own standing pose) --
    # but ONLY once already near standing height, same phase-gate mechanism as knee_bend_phase above (see
    # final_pose_phase_penalty's own docstring for why this can't be active earlier -- it would otherwise
    # pay the robot for simply not moving, since deviation is ~0 at the very start of every episode).
    # User's own insight: height/orientation/stability_hold all check pelvis position and torso tilt, but
    # none of them check whether the actual JOINT CONFIGURATION looks like a correct, stable stand (legs
    # could be crossed, knees asymmetric, arms still out, while height+tilt are still satisfied). Overlaps
    # partially with knee_bend_phase's own penalty side (both discourage bent knees near standing) --
    # harmless redundancy, same direction, not a conflicting signal. Weight is a first-pass placeholder.
    final_pose_phase = RewTerm(
        func=asimov1_mdp.final_pose_phase_penalty,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "standing_height": STANDING_HEIGHT,
            "transition_width": 0.15,
        },
    )

    # fixes the torso-twist the user observed live once robots reach the sitting waypoint (run 7: improved
    # over the run from fully-backwards to half-cocked-sideways, never fully resolved). `final_pose_phase`
    # above doesn't help -- it's gated on proximity to STANDING_HEIGHT, effectively off at SIT_HEIGHT. See
    # `sit_torso_alignment_penalty`'s own docstring for why this reuses `default_joint_pos` as the target
    # (no new seated-pose array needed) with a different joint subset + a band gate instead of a one-sided
    # ramp. Weight is a first-pass placeholder, same order of magnitude as final_pose_phase.
    sit_torso_alignment = RewTerm(
        func=asimov1_mdp.sit_torso_alignment_penalty,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=SIT_TORSO_ALIGNMENT_JOINTS),
            "sit_height": SIT_HEIGHT,
            "band_width": 0.15,
        },
    )

    # continuous (never hard-terminating) dwell penalty -- added after run 11 reliably reached sitting fast
    # per the user's explicit request: "tighten up the penalty for sitting too long... or staying on their
    # backs for too long." Covers both in one mechanism -- see SitDwellPenalty's own docstring.
    # weight raised -1.5 -> -4.0 (matching upper_body_ground's own successfully-dominant precedent) after
    # run 12's first-pass -1.5 weight was confirmed TOO WEAK: over iterations 876->996, sit_height_tracking
    # (1.04->1.21) and orientation_upright (0.81->1.13) both kept climbing -- the policy getting BETTER at
    # sitting -- while kneel_height_tracking stayed flat (0.037->0.036) and the dwell penalty itself barely
    # moved (-1.237->-1.246), confirmed live by the user ("more and more are local optimum-ing into a seated
    # position"). At -1.5, the net combined reward from sitting well (sit_height_tracking + orientation_
    # upright - sit_torso_alignment =~ +1.66 at those numbers) still comfortably outweighed the dwell cost,
    # so there was no real pressure to risk the harder climb. -4.0 is sized to clearly exceed that net
    # sitting reward once the penalty ramps to its ceiling (max_penalty=1.0 * -4.0 = -4.0, vs a ~1.5-1.7
    # sitting payout) -- same reasoning already validated for `upper_body_ground_penalty` earlier this
    # session: a dominant PENALTY doesn't carry the same farming/lock-in risk a dominant REWARD does.
    sit_dwell_penalty = RewTerm(
        func=asimov1_mdp.SitDwellPenalty,
        weight=-4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_height": 0.9 * STANDING_HEIGHT,
            "min_progress": 0.03,
            "window_steps": 120,
            "grace_steps": 60,
            "max_penalty": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    """`time_out` ONLY -- see module docstring for why every other asimov1 task's termination set (pelvis
    contact, collapsed height, bad orientation, feet crossed/too-wide) would fire every step here by
    design and is deliberately not reused. TWO different hard terminations were tried and deliberately NOT
    added here -- `StalledRecoveryTermination` and `HeadGroundTermination`, both still fully written in
    mdp.py for whenever they make more sense. See the comment below for why.

    `HeadGroundTermination` (a hard termination for head/shoulder-ground contact, added after the graded
    `upper_body_ground_penalty` reward alone wasn't stopping the A-pose) was tried TWICE and failed both
    times: at grace_period=2.0/min_height=0.3, episode length stayed pinned at exactly the grace-period
    boundary for an entire 300-iteration verification run, zero escapes; loosened to grace_period=6.0
    (half the episode)/min_height=0.15 (a much easier bar -- just lift off the ground at all), SAME result,
    zero escapes. Both share the root problem `StalledRecoveryTermination` already hit: a hard pass/fail
    check applied to a policy that hasn't yet learned the underlying skill will universally fail,
    regardless of how generous the parameters are, because there's no learned behavior yet to pass it with.
    Resolution: dropped the termination, put the "stop this behavior harder" burden back on
    `upper_body_ground_penalty` (continuous, already proven safe, weight raised -2.0 -> -4.0 -- see that
    term's own comment) instead of chasing a third parameter guess on an approach that's now failed twice.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # NOT ADDED (yet) -- StalledRecoveryTermination exists in mdp.py but is deliberately left OUT of this
    # task for now. Tried window_time=3.0/min_height_gain=0.05, then loosened to window_time=6.0/
    # min_height_gain=0.02 -- BOTH fired on 100% of episodes in the smoke test, regardless of how generous
    # the parameters were. Root cause isn't the threshold: for a random/untrained policy, net height change
    # over ANY window is reliably ~0 (gravity settles things down, nothing yet drives it up), so a "did you
    # make progress" check fails universally before any learned climbing behavior exists to make progress
    # WITH -- the same class of mistake as the premature curriculum ramp fixed earlier this session
    # (clamping down before the policy has anything to work with). InsufficientProgressTermination (walking
    # task, the mechanism this mirrors) was itself added to an ALREADY-competent policy exhibiting a
    # specific plateau, not from iteration 0 on a random one -- this needs the same treatment: introduce it
    # on a RESUME once this run's own reward/termination changes show real baseline climbing behavior, not
    # from scratch.


@configclass
class CurriculumCfg:
    """HumanUP's two-phase curriculum, implemented via the stock `modify_reward_weight` term (already
    proven in lift_env_cfg.py/reach_env_cfg.py -- zero new/unproven code).

    num_steps raised 6000 -> 200000 after real-run evidence showed 6000 was far too early: at
    num_steps_per_env=32, that threshold fires at iteration ~187 (confirmed directly in the log -- the
    ramp actually fired at iteration 203) out of a 1500-iteration run, i.e. only ~13.5% in. The robots
    spent the other ~86.5% of that run under tightened torque/action-rate/contact-force penalties before
    any of them had likely ever discovered a path past a tripod/"A-pose" into standing -- height_tracking's
    graded exp kernel already pays real partial credit for ANY elevated pose, so once regularization
    clamped down, the harder torque-heavy burst needed to launch from tripod into a full stand became
    *more* expensive to attempt than just holding the free partial-credit pose, visibly observed live as
    robots stalling in an A-pose and never progressing further. 200000 > 1500*32=48000 (this run's total
    step count), so Phase 1's relaxed constraints now stay active for the ENTIRE run -- this is a direct
    test of whether standing is achievable at all under permissive conditions, before reintroducing any
    phase transition (fixed-later or success-rate-gated) once that's confirmed."""

    ramp_action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -0.03, "num_steps": 200000},
    )
    ramp_dof_torques = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "dof_torques", "weight": -1.0e-5, "num_steps": 200000},
    )
    ramp_contact_force_limit = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "contact_force_limit", "weight": -5.0e-4, "num_steps": 200000},
    )

    # assist-force curriculum (see asimov1_mdp.AssistCurriculum/apply_assist_force docstrings) -- NEW this
    # run, ported from OpenHorizon's HoST-inspired getup policy (openhorizon-labs/asimov-v1-policies), which
    # reaches 100% sim success using exactly this mechanism. Unlike the three ramps above (step-count-gated,
    # effectively disabled this run), this one is success-RATE-gated: the assist decays only once the
    # measured standing-success rate clears `ASSIST_SUCCESS_THRESHOLD`, same as OpenHorizon's own design.
    assist = CurrTerm(
        func=asimov1_mdp.AssistCurriculum,
        params={
            "initial_scale": ASSIST_INITIAL_SCALE,
            "decay_step": ASSIST_DECAY_STEP,
            "success_threshold": ASSIST_SUCCESS_THRESHOLD,
            "window_steps": ASSIST_WINDOW_STEPS,
            "min_pelvis_height": 0.9 * STANDING_HEIGHT,
            "min_upright": 0.9,
            "feet_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            ),
        },
    )


@configclass
class ASIMOV1FallRecoveryEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the asimov-1 fall-recovery (getup) environment."""

    scene: FallRecoverySceneCfg = FallRecoverySceneCfg(num_envs=256, env_spacing=3.0, clone_in_fabric=True)
    observations: Asimov1FallRecoveryObservationsCfg = Asimov1FallRecoveryObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 2
        self.episode_length_s = 12.0
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physics = Asimov1PhysicsCfg()
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0

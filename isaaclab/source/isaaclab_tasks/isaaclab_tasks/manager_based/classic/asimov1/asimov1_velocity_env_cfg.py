# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity-commanded locomotion task for asimov-1 -- the pivot from the (now solid, 73% time_out)
stand-and-balance task in `asimov1_env_cfg.py` toward actual walking, mirroring v1x1's own progression
(single-vector walk -> omni-directional velocity-command task).

Co-located in this same `classic/asimov1/` folder rather than moved to `locomotion/velocity/config/asimov1/`
(Isaac Lab's own idiomatic location for this task style) for the same reason v1x1's velocity task stayed in
`classic/v1x1/`: this task reuses the standing task's scene/actions/events verbatim (`MySceneCfg`,
`Asimov1ContactSensorCfg`, `Asimov1PhysicsCfg`, `ActionsCfg`, `EventCfg` -- all imported, not duplicated) and
most of its custom `mdp.py` terms (`feet_separation_l2`, `feet_stance_width_l2`, `feet_crossed`,
`feet_too_wide`, `recovery_step_bonus`, `body_vertical_l2`) are still relevant to a walking gait, so splitting
into a separate directory would just duplicate code or force cross-directory imports for no benefit.

FORWARD-AXIS CHECK (done fresh for this task, not assumed): confirmed from the URDF's own foot collision
geometry (`left_ankle_roll_link`'s 4 collision spheres: heel at x=-0.045, toe at x=+0.122 -- the foot
extends much further in +X than -X) that asimov-1's real forward is local **+X**, matching Isaac Lab's own
H1/G1 convention. This is DIFFERENT from v1x1, whose real forward was local +Y (a quirk of that robot's own
mount orientation) -- asimov-1 needs no axis remapping: `lin_vel_x` means forward/back exactly as it does for
every stock velocity task, `lin_vel_y` means left/right strafe, `ang_vel_z` means turn rate. Do not assume
this transfers to any future robot without re-checking the same way.

FRESH START REQUIRED: cannot resume from any `Isaac-Asimov1-Stand-v0` checkpoint (e.g. `model_8995.pt`) --
the observation space changes (new `velocity_commands` term added), so the policy network's input layer
shape differs and `runner.load()` would fail on a shape mismatch. This mirrors the exact same constraint
v1x1 hit for its own velocity-command pivot.

REWARD-SHAPE CHANGES FROM THE STANDING TASK (the actual pivot, not a copy-paste):
- DROPPED entirely: `lin_vel_xy`/`ang_vel`/`heading_hold_bonus` -- these actively penalize ANY base motion
  or heading change, which would directly fight a walking gait by design, not just need re-tuning.
- DROPPED `joint_deviation_legs`/`joint_deviation_feet` -- these penalized hip_pitch/knee/ankle deviation
  from zero, exactly the joints that MUST move for a stride. Isaac Lab's own H1 reward shape
  (`locomotion/velocity/config/h1/rough_env_cfg.py::H1Rewards`) makes the identical choice: it only penalizes
  deviation on hip_yaw/hip_roll/arms/torso, never hip_pitch/knee/ankle. `hip_torso_rotation` below is
  extended to cover hip_roll (previously only hip_yaw+waist_yaw) to keep that non-propulsive-axis damping.
- ADDED `track_lin_vel_xy`/`track_ang_vel_z` (yaw-frame/world-frame variants, same functions H1 itself uses --
  more robust to roll/pitch sloshing than the plain body-frame versions), `lin_vel_z`/`ang_vel_xy` (standard
  companion penalties against vertical bounce / excess roll-pitch rate), `feet_air_time` (rewards alternating
  single-stance steps, zero reward at zero command), `feet_slide` (penalize feet dragging while in contact),
  `stand_still_joints` (only penalizes hip_pitch/knee/ankle deviation when the command is near-zero, so
  standing-still quality is preserved without fighting a nonzero-command stride) -- all pulled unmodified
  from `isaaclab_tasks.manager_based.locomotion.velocity.mdp`, aliased `loco_mdp` below.
- `shin_vertical`/`torso_vertical` weights cut roughly 3x (-1.0/-0.9 -> -0.3/-0.3) as a FIRST GUESS, not a
  measurement -- a real stride necessarily tilts the shin during swing phase and the torso often leans
  slightly forward, so the standing task's full weight would likely fight normal walking. Watch live and
  adjust; this is explicitly the kind of placeholder this project always flags for a follow-up tuning pass.
- REMOVED `foot_airborne` termination entirely -- "any sustained foot-lift is bad" is correct for standing
  and actively wrong for walking (a real swing phase needs meaningful air time). Noted as the expected
  removal in the asimov1 project memory's "STRATEGIC DIRECTION" section before this file was written.
- KEPT unchanged: `termination_penalty` (-1500, still the critical anti-reward-hacking fix),
  `feet_separation`/`feet_stance_width`/`feet_crossed`/`feet_too_wide` (anatomical feet-crossing/spreading
  guards, still relevant mid-stride), `recovery_step_bonus` (only engages past tilt_threshold=0.15, well
  beyond a normal walking lean, so it should stay dormant during ordinary gait and still fire on real
  stumbles), `base_height`, `joint_deviation_arms`/`joint_deviation_waist`, smoothness penalties.
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loco_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp import foot_clearance_reward
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from . import mdp as asimov1_mdp
from .asimov1_env_cfg import (
    ActionsCfg,
    Asimov1PhysicsCfg,
    EventCfg as _BaseEventCfg,
    MySceneCfg,
)

from isaaclab.envs import ManagerBasedRLEnvCfg


@configclass
class EventCfg(_BaseEventCfg):
    """Same reset_base/push_robot as the standing task, but with real joint-pose randomization at reset --
    the standing task's exact-default-pose reset (position_range=(1.0, 1.0)) made sense for a from-scratch
    standing policy (clean, repeatable pose to learn to hold), but for a walking task it meant the policy
    NEVER had to actively re-stabilize from a varied starting state -- it could always fall back on "a small
    correction routine around this one familiar pose" and relax into it once no longer in danger. Compared
    against the Isaac Lab STANDARD recipe (locomotion/velocity/velocity_env_cfg.py's own EventsCfg, which
    every legged-robot task inherits): position_range=(0.5, 1.5) there. (0.7, 1.3) here as a less aggressive
    first step (asimov1 is still a novice walker, unlike the mature quadrupeds that recipe was tuned on)."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.7, 1.3), "velocity_range": (0.0, 0.0)},
    )

    # push_robot's forward-biased x-push REVERTED back to symmetric for the SECOND fresh restart, same
    # reasoning as the command range revert above -- isolate the directional-bug fix as the one real change
    # from the first fresh restart's starting config, see if it alone resolves the asymmetry before
    # reintroducing any bias mechanism.
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # Conservative first-pass ranges -- this policy has never walked before (only ever learned to stand), so
    # starting small and raising later mirrors how v1x1 itself started gentle before scaling up to its proven
    # ~0.5-0.6 m/s committed-gait speed. rel_standing_envs=0.2 (not the default 0.0) keeps a meaningful chunk
    # of episodes commanded to stand still, so the already-solid standing skill isn't crowded out while
    # walking is still being learned.
    # debug_vis=True -- defaults to False on CommandTermCfg (confirmed by reading manager_term_cfg.py), which
    # is why the goal/current velocity arrows (the same green/blue markers v1x1's velocity task showed) were
    # silently absent from the viewport despite the UniformVelocityCommandCfg's own
    # goal_vel_visualizer_cfg/current_vel_visualizer_cfg being defined -- those only get wired up when
    # debug_vis is explicitly turned on.
    # Ranges WIDENED live mid-session (Β±0.3/Β±0.2/Β±0.5 -> Β±0.6/Β±0.4/Β±0.8) after comparing against the shared
    # LocomotionVelocityRoughEnvCfg base class (locomotion/velocity/velocity_env_cfg.py) that EVERY Isaac Lab
    # legged robot task inherits from -- its own default command range is Β±1.0/Β±1.0/Β±1.0 paired with
    # std=0.5. Our original Β±0.3/Β±0.2 was too narrow RELATIVE TO std=0.5 for standing-still to ever be
    # meaningfully punished (see track_lin_vel_xy's own comment below) -- widening the range is the
    # standard-recipe fix, not just tightening std in isolation on an unusually narrow range.
    # lin_vel_x's forward-only practice bias ((0.0, 0.6)) REVERTED back to symmetric for the FRESH RESTART
    # (iteration 0) -- that bias was a targeted fix for a specific checkpoint's learned backward/forward
    # asymmetry (a path-dependent habit from 20 runs of accumulated training), not a structural property
    # worth baking into a brand-new policy that hasn't developed that habit yet. A fresh policy should
    # learn forward and backward symmetrically given correct reward shape (std/range/weights all now
    # fixed) -- revisit the forward-only bias again later ONLY if the same asymmetry re-emerges.
    #
    # lin_vel_x's mostly-forward bias ((-0.1, 0.6)) REVERTED back to symmetric for the SECOND fresh restart
    # -- that bias made things measurably worse (insufficient_progress 88.9%->93.5%, episode length
    # 485->252), the same destabilizing signature as the FIRST forward-only attempt, before we'd found that
    # InsufficientProgressTermination had a real directional-measurement bug the whole time. With that bug
    # fixed, the SECOND restart tried clean symmetric commands + correct measurement -- 4 runs later,
    # insufficient_progress plateaued flat (93.1%->88.2%->87.3%) despite balance still improving, i.e. the
    # bug fix alone did NOT resolve the forward deficit.
    #
    # lin_vel_x/lin_vel_y CHANGED to forward+turn-only for the THIRD fresh restart (FRESH3), directly
    # matching Unitree H1's AND G1's own real, validated Isaac Lab recipes -- neither official biped task
    # ever trains on backward or lateral commands AT ALL, only forward speed + turning. That recipe, plus
    # the actuator-stiffness rebalance and action_rate fix from the same comparison, produced the best
    # results this entire project by a wide margin (FRESH3 runs 1-4: insufficient_progress 44.75%->1.2%->
    # 19.1%->18.5%, error_vel_xy down to 0.089, time_out up to 79%, genuinely natural-looking forward gait
    # per live user observation -- "zombie shuffling" -> "nearly perfect"). See project memory for full detail.
    #
    # lin_vel_x/lin_vel_y WIDENED to full omni-directional (Β±0.6/Β±0.4) FIRST -- tried as a direct resume
    # from FRESH3's model_5996.pt ("OMNI_run1"). Result: NOT a quick shock-then-recover like the
    # action_scale change -- insufficient_progress climbed monotonically the ENTIRE run (27%->63%->69%,
    # mean_reward 96.4->24.4) with no deceleration by the end, while collapsed/orientation stayed LOW the
    # whole time (balance itself never broke -- it just wasn't executing non-forward commands, confirmed
    # live: "some of them are ignoring their commanded heading and just going straight anyways"). Pattern-
    # matched against this project's own history: this shape (monotonic climb, no plateau, full run) matches
    # the TWO prior "too big a jump" cases (the old lineage's forward-only command bias, and the
    # min_speed_fraction=0.3 termination attempt) far better than the quick-recovery cases (action_scale),
    # whose common thread was recalibrating an ALREADY-KNOWN skill to new numbers, not learning a genuinely
    # new behavior region from a conservative entropy_coef=0.003 policy that's confidently converged on
    # "just walk forward."
    #
    # CURRICULUM APPROACH instead, per the same "narrow bite first, widen later" pattern that worked for
    # the forward-only range itself earlier in this project (Β±0.3/Β±0.2 -> Β±0.6/Β±0.4 happened gradually, not
    # in one jump). First step: lin_vel_x=(-0.2,0.6)/lin_vel_y=(-0.15,0.15), resumed from FRESH3's
    # model_5996.pt (the last known-good checkpoint, pre-regression). Result over 2 full runs (~3000 iters):
    # found a STABLE plateau almost immediately (insufficient_progress ~51-58%, error_vel_xy ~0.11-0.15,
    # collapsed <2%) and held there -- no further degradation (unlike the full-range jump's monotonic climb
    # to 69%) but also no further improvement between the two runs, i.e. this step is "solid," time to widen.
    #
    # IMPORTANT CONTEXT (user's own stated goal, not just a curriculum exercise): full omnidirectional is
    # the ACTUAL end target, not a stretch goal to defer -- asimov-1 needs to eventually navigate tight
    # spaces/hallways and do household chores on real hardware, which requires fluid forward/backward/
    # lateral combination, not just "walk in a line and turn" (H1/G1/Cassie's own narrower recipes). Checked
    # Digit (Agility Robotics, also in this same locomotion/velocity/config tree) -- it DOES train full
    # omnidirectional (lin_vel_x=(-0.8,0.8), lin_vel_y=(-0.5,0.5)), proving this is achievable in this exact
    # framework, not a dead end -- just a harder curriculum than H1/G1's forward-only recipe, which is why
    # 3 of 4 other reference bipeds avoid it. Widen toward THAT range over several more steps, not settling
    # for forward+turn-only.
    #
    # SECOND widen step: lin_vel_x=(-0.4,0.6), lin_vel_y=(-0.3,0.3) -- both dimensions widened together,
    # roughly halfway from step 1 toward Digit's full range. Resuming from step 1's own run2 checkpoint (the
    # stable plateau), not rolling back -- step 1 didn't regress, it plateaued, which is the "ready to widen"
    # signal, not the "roll back" signal. Result: run1 surfaced hip_yaw_limit climbing (2.7%->5.1%->5.2%->
    # 12.3%) specifically as lateral/backward commands were introduced -- fixed by adding Menlo's own
    # `angular_momentum`/stronger `ang_vel_xy` lateral-stability terms (see RewardsCfg below), validated
    # across 3 more runs: hip_yaw_limit settled to a stable 0.8-3.7% band, reward recovered to match/beat the
    # pre-fix baseline by the final run. Checkpoint `2026-10-10_22-24-00/model_14990.pt` is the validated
    # result of this whole step.
    #
    # Also found MENLO'S OWN real command envelope for this exact robot (their published training-env docs,
    # not inference from another robot): lin_vel_x=(-0.8,0.8), lin_vel_y=(-0.6,0.6), ang_vel_z=(-0.6,0.6) --
    # this supersedes the earlier Digit-based approximation as the actual target to widen toward. Note
    # ang_vel_z is already WIDER here (Β±0.8) than Menlo's own Β±0.6 -- not widening that dimension further.
    #
    # THIRD widen step: lin_vel_x=(-0.6,0.8), lin_vel_y=(-0.45,0.45) -- both dimensions widened together
    # again, roughly halfway from step 2 toward Menlo's real Β±0.8/Β±0.6 target. Resumed from step 2's
    # validated final checkpoint (model_14990.pt). Result over 4 runs: run1/run2 showed hip_yaw_limit
    # elevated/noisy (12.5%->14.5%), run3/run4 settled into a stable low band (6.1%->8.4%) while
    # reward/error_vel_xy/time_out ALL kept climbing to new bests simultaneously (reward 39.9->52.3,
    # error_vel_xy 0.271->0.141, time_out 25%->37.4%) -- multi-metric confirmation of genuine settling, not
    # a lucky single run. Checkpoint `2026-10-11_19-38-XX/model_20986.pt` (run4's finish) is the validated
    # result of this whole step, the best checkpoint in the entire omnidirectional lineage so far.
    #
    # FOURTH widen step (this one): lin_vel_x=(-0.7,0.8), lin_vel_y=(-0.55,0.55) -- roughly halfway from
    # step 3 toward Menlo's full Β±0.8/Β±0.6 target. ang_vel_z left UNCHANGED at Β±0.8 (already wider than
    # Menlo's own Β±0.6 target, confirmed above -- no need to touch it). This step is close enough to the
    # final target that the NEXT widen after this one could plausibly go straight to the full Β±0.8/Β±0.6.
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        heading_command=False,
        debug_vis=True,
        rel_standing_envs=0.2,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.7, 0.8),
            lin_vel_y=(-0.55, 0.55),
            ang_vel_z=(-0.8, 0.8),
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        # the one genuinely new observation this pivot needs -- exposes the currently sampled command so the
        # policy can condition its behavior on what it's actually being asked to do.
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class Asimov1VelocityObservationsCfg(PresetCfg):
    default: ObservationsCfg = ObservationsCfg()
    physx: ObservationsCfg = ObservationsCfg()
    newton_mjwarp: ObservationsCfg = ObservationsCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Stay alive bonus + the anti-reward-hacking fix -- see asimov1_env_cfg.py's RewardsCfg docstring/
    # comments for the full saga behind -1500 (that value is CORRECT for the standing task, which still
    # uses it unchanged).
    # LOWERED -1500 -> -300 for the velocity/AMP task specifically, live mid-session, per a first-principles
    # argument: with gamma=0.99, PPO's effective planning horizon is ~100 steps (1/(1-gamma)), over which
    # the max accumulable `alive` reward is only ~63 (discounted sum of 1.0/step). A -1500 one-time penalty,
    # even 100 steps out, still has a discounted present value of roughly -550 -- dwarfing anything any
    # combination of track_lin_vel_xy/foot_clearance/etc. could realistically accumulate in that same
    # window. No amount of buffing the "go" rewards (tried 3 separate times this session: 1.5->3.0->4.5->
    # 6.0) could ever out-compete a stick that disproportionate -- it was pure risk-aversion overwhelming
    # every carrot. -1500 was tuned for the STANDING task to kill a specific reward-hacking exploit
    # (deliberately crossing feet to end episodes cheaply) -- that exploit has since been independently
    # fixed for this task (feet_crossed/feet_too_wide loosened based on real gait data, holding at 0% across
    # many consecutive runs), so this task no longer needs anywhere near that much deterrent. -300 is a
    # first guess at "still a real deterrent, no longer an overwhelming one" -- watch live and adjust.
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-300.0)
    # (2) Velocity TRACKING, not velocity SUPPRESSION -- the actual pivot. Yaw-frame/world-frame variants
    # (same functions H1 itself uses) rather than the plain body-frame versions -- more robust to roll/pitch
    # sloshing during a real stride.
    # Weight raised 1.5 -> 3.0 live mid-session per explicit observation: the policy was visibly ATTEMPTING
    # steps but staying overly conservative -- plausible explanation is that termination_penalty (-1500,
    # one-time) so dominates the value function that any step attempt risky enough to threaten a fall isn't
    # worth it when the per-step tracking payout is only a fraction of a point. Doubling the tracking reward
    # raises the upside of successfully moving without touching the termination penalty itself (a separate,
    # deliberate choice -- the penalty is still the right size for ITS job of preventing reward-hacking via
    # cheap terminations, this is a different lever: making genuine forward progress worth more risk).
    # Weight raised 3.0 -> 4.5 -> 6.0 live mid-session (AMP task), two separate bumps for the same
    # underlying symptom: balance/foot-lift are now both solid (foot_clearance hit 96% of its own ceiling,
    # time_out jumped to 69% after the 4.5 bump + foot-separation fixes) but the robots were STILL visually
    # just balancing in place, not translating -- the exp-tracking kernel's forgiving partial credit
    # (error_vel_xy=0.227 against a 0.5 std already earns ~81% of the per-step reward) let "mostly
    # stationary with a small residual velocity" look deceptively good on paper without real net progress.
    # Paired this round with the new foot_stuck_penalty above, which more directly targets the actual
    # "not stepping" behavior rather than relying on the tracking reward alone to force it.
    #
    # CORRECTION, several runs later: raising WEIGHT was the wrong lever the whole time -- it scales the
    # reward curve uniformly, it doesn't change its SHAPE, so "stand still" vs "track perfectly" stayed in
    # roughly the same ratio no matter how high the weight went. Verified directly: with std=0.5 against
    # this task's own command range (lin_vel_x +-0.3, lin_vel_y +-0.2), standing COMPLETELY still scores
    # exp(-((0.15)^2)/0.25)=91% of max reward for a modest command, and exp(-(0.3^2+0.2^2)/0.25)=60% of max
    # even at the hardest command -- three separate weight raises (1.5->3.0->4.5->6.0) could never have
    # fixed that, since 91%/60% vs 100% stays the same ratio regardless of the multiplier in front. std
    # TIGHTENED 0.5 -> 0.15 instead: at std=0.15, those same standing-still cases drop to 37%/0.3% of max --
    # a real forcing function for the first time, rather than a weight that was never doing anything.
    # Weight raised 6.0 -> 9.0 live mid-session, this time a CORRECTLY-targeted raise -- now that the
    # kernel's shape actually discriminates between moving and not, a bigger weight genuinely increases the
    # pull toward real tracking relative to other terms, rather than just rescaling an already-flat curve.
    #
    # std LOOSENED 0.15 -> 0.25 live mid-session, alongside widening CommandsCfg's own ranges above: 0.15
    # was overcompensating for a command range (+-0.3/+-0.2) that was unusually narrow relative to the
    # Isaac Lab STANDARD recipe (std=0.5 paired with +-1.0/+-1.0 ranges, confirmed by reading the shared
    # LocomotionVelocityRoughEnvCfg base class every legged-robot task inherits from). Now that the range
    # itself is wider (+-0.6/+-0.4), std=0.25 still gives real discrimination (standing still against the
    # new range's typical command scores ~exp(-0.36/0.0625)=0.3% of max) without the needle-sharp, possibly
    # harder-to-learn-from shape std=0.15 created on top of an already-narrow range.
    # Weights reset 9.0/0.75 -> 2.0/2.0 for the FRESH RESTART (iteration 0) per the user's own research
    # into Menlo's actual published weights for this exact robot: track_lin_vel=+2.0, track_ang_vel=+2.0
    # (same magnitude for both, unlike our own asymmetric 9.0/0.75). Our own 9.0 was arguably inflated to
    # compensate for the OLD too-narrow command range/too-tight std combo (see the correction comments
    # above) -- now that std=0.25 + range +-0.6/+-0.4 are independently well-reasoned (not copied from
    # Menlo, derived from the generic Isaac Lab standard recipe), a smaller weight like Menlo's own 2.0
    # should still provide real signal without as aggressively dominating the rest of the reward budget.
    # Weight raised 2.0 -> 4.0 after the fresh restart's 3rd chunk: balance fundamentals solidified nicely
    # (collapsed+orientation down to ~15.5% combined, best yet for this lineage) while insufficient_progress
    # leveled off around 81-84% rather than still climbing -- the same "balance is solid, now nudge the
    # tracking incentive" moment that eventually worked in the old 20-run lineage.
    track_lin_vel_xy = RewTerm(
        func=loco_mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=4.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    track_ang_vel_z = RewTerm(
        func=loco_mdp.track_ang_vel_z_world_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # weight raised -0.05 -> -0.08 to match Menlo's own published `body_ang_vel` weight for this exact robot
    # (their reward-design docs, pasted by user) -- treated as the same physical quantity (roll/pitch base
    # angular velocity) rather than adding a separate duplicate term under their name.
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.08)
    # NEW -- Menlo's `angular_momentum` term, added specifically because "Asimov has a narrower stance...
    # increases lateral balance sensitivity." Added live mid-session when hip_yaw_limit was observed
    # climbing across the omni-directional curriculum widening -- see whole_body_angular_momentum_l2's
    # docstring in mdp.py for the full reasoning and the orbital-only simplification used. Weight -0.03 is
    # Menlo's own published value, not guessed.
    angular_momentum = RewTerm(func=asimov1_mdp.whole_body_angular_momentum_l2, weight=-0.03)
    # (3) Stay near the standing height estimate -- unchanged.
    base_height = RewTerm(func=mdp.base_height_l2, weight=-1.0, params={"target_height": 0.63})
    # ADDED for the fresh restart: an explicit "upright" reward -- Menlo's own recipe includes one
    # (reported weight +1.0) that this task never had an equivalent of. We only had INDIRECT proxies
    # (shin_vertical/torso_vertical/base_height), nothing checking the root/pelvis orientation itself the
    # way the standing task's own `flat_orientation` does. Reused that same stock function here, at the
    # SAME SIGN CONVENTION already used everywhere else in this project (a penalty on tilt, weight
    # negative) rather than guessing Menlo's own reward-vs-penalty framing from a bare "+1.0" number alone
    # -- `flat_orientation_l2` returns a tilt MAGNITUDE (0 when upright), so a positive weight on it would
    # backwards-reward tilting MORE, not less.
    # weight SOFTENED -1.0 -> -0.3 live mid-session -- same reasoning as torso_vertical's new dead-zone
    # above (user's own question: this was punishing a forward lean that real forward propulsion may
    # genuinely need). flat_orientation_l2 is a stock function with no tolerance param to add a dead-zone
    # to directly, so softening the weight is the available lever here instead.
    #
    # weight RAISED -0.3 -> -5.0 (Menlo's own exact published "orientation" weight for this robot) once the
    # omnidirectional curriculum (steps 1-4) was fully validated and the focus shifted to gait QUALITY
    # (smoothness/stability) rather than command-range coverage -- user's own explicit ask: "which [lever]
    # will get us a more accurate and cleaner/smoother omni-gait" between widening further vs. tuning,
    # chose tuning. Magnitude-checked SAFE before applying (unlike action_rate below): logged
    # Episode_Reward/flat_orientation at the old weight was only -0.0014 to -0.0041, i.e. raw unweighted
    # magnitude ~0.005-0.014 -- at -5.0 that becomes roughly -0.025 to -0.07, still small relative to
    # track_lin_vel_xy's own ~1.9-2.3 weighted contribution. Direct copy of Menlo's value is appropriate
    # here (unlike action_rate, where the raw magnitudes were NOT comparable -- see that term's own comment).
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    # hip_torso_rotation REMOVED live mid-session -- this directly answers the user's own question
    # ("I wonder if we have conflicting rewards for staying in place and going in the right direction?").
    # It penalized hip_yaw/hip_roll/waist_yaw deviation from ZERO, correct for the standing task where any
    # such rotation genuinely was an unwanted habit, but measured directly against the AMP reference clip's
    # own real walking gait: left_hip_yaw averages -0.104 rad, right_hip_yaw +0.092, waist_yaw +0.036 --
    # completely normal counter-rotation for balance during a real stride, NOT a mistake. At weight -1.5,
    # this was costing ~-0.455 EVERY STEP just for performing the reference gait's own correct motion -- a
    # literal reward-vs-reward conflict, since these same joints are exactly what the AMP discriminator is
    # simultaneously pulling toward via joint_pos matching against this identical reference data. Removing
    # this hand-shaped zero-target penalty loses no real guidance: AMP's own matching already encodes the
    # CORRECT non-zero target for these joints, which this term was fighting.
    # (5) Gait-shaping rewards, pulled from Isaac Lab's own biped locomotion library (loco_mdp) rather than
    # reinvented -- feet_air_time rewards alternating single-stance steps (zero reward at zero command, so it
    # doesn't fight the rel_standing_envs fraction), feet_slide penalizes a planted foot dragging.
    # Weight raised 0.25 -> 0.5 live mid-session per explicit observation: "a few baby steps here and there
    # but still no walking cyclically" -- this is the one reward that directly rewards the ALTERNATING
    # pattern itself (not just net displacement like track_lin_vel_xy), so it needed more relative weight to
    # push toward sustained stepping rather than isolated ones. Paired with tightening foot_stuck below.
    feet_air_time = RewTerm(
        func=loco_mdp.feet_air_time_positive_biped,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "threshold": 0.4,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            ),
        },
    )
    feet_slide = RewTerm(
        func=loco_mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
        },
    )
    # (5a) Added live mid-session per explicit observation: balance is solid now (foot_clearance/
    # track_lin_vel_xy both engaging well per the log) but the robots were visually just balancing in place,
    # not translating. `feet_air_time_positive_biped` only rewards GOOD alternation when it happens -- it
    # never penalizes a foot that simply never lifts, so "shift weight slightly, never really step" was an
    # unpunished strategy as long as balance/tracking partial-credit stayed decent under the forgiving exp
    # kernel. See mdp.py's foot_stuck_penalty docstring for the max_contact_time derivation (grounded in
    # the AMP reference clip's own measured stance-phase durations, not guessed).
    # max_contact_time TIGHTENED 1.0 -> 0.75 live mid-session per explicit observation: "a few baby steps
    # here and there but still no walking cyclically" -- 1.0s was loose enough that the policy could take
    # one isolated step, stand comfortably for just under a second, take another isolated step, and never
    # once trigger this penalty, satisfying it without ever committing to continuous cyclic stepping. The
    # real reference gait's own stance phases are only 0.62-0.68s (median/90th pct) -- 0.75s sits just above
    # that real range, tight enough to make "stand between occasional steps" actually cost something.
    foot_stuck = RewTerm(
        func=asimov1_mdp.foot_stuck_penalty,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            ),
            "command_name": "base_velocity",
            "max_contact_time": 0.75,
        },
    )
    # (5a2) Added live mid-session per the user's own suggestion ("penalize the steps being too fast...
    # slow/smooth out the steps") -- tippy-tapping persisted (reduced, not eliminated) even after
    # foot_stuck above and a higher track_lin_vel_xy weight both landed. Mirror image of foot_stuck: this
    # penalizes a foot that lifted and landed again too QUICKLY instead of too slowly. See mdp.py's
    # foot_swing_too_short_penalty docstring for the min_swing_time=0.25 derivation (grounded in the AMP
    # reference clip's own measured swing-phase durations).
    foot_swing_too_short = RewTerm(
        func=asimov1_mdp.foot_swing_too_short_penalty,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            ),
            "command_name": "base_velocity",
            "min_swing_time": 0.25,
        },
    )
    # (5b) Added live mid-session per explicit observation ("maybe if they lifted their feet more they could
    # balance and possibly walk better") -- `feet_air_time_positive_biped` above only rewards single-stance
    # DURATION, it has no notion of how HIGH the swinging foot actually gets off the ground, so nothing in
    # this task was previously pushing for real ground clearance during a stride. Reused verbatim from
    # Isaac Lab's own Spot quadruped task (`locomotion/velocity/config/spot/mdp/rewards.py`) -- scales a
    # foot-height-from-target error by the foot's own horizontal velocity (via tanh), so it only grades
    # clearance while a foot is actually swinging, not while planted. target_height=0.08 is an ESTIMATE
    # (ankle_roll_link's own collision geometry sits ~0.029m above the sole at rest, so this targets ~5cm of
    # real lift above a flat foot) -- flagged for visual tuning like every other numeric placeholder in this
    # project, not measured. std/tanh_mult copied from Spot's own values as a starting point.
    # Weight raised 0.5 -> 1.5 live mid-session per explicit observation that feet still weren't lifting
    # enough despite this term already being active -- confirmed in the log, foot_clearance was only
    # earning ~0.052 of its own 0.5 ceiling (~10%), a weak fraction, so the incentive needed to be bigger
    # relative to the rest of the reward shape, not redefined (target_height/std/tanh_mult left untouched
    # to isolate this as a single-variable change).
    foot_clearance = RewTerm(
        func=foot_clearance_reward,
        weight=1.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            ),
            "target_height": 0.08,
            "std": 0.05,
            "tanh_mult": 2.0,
        },
    )
    # (6) Only penalizes leg-joint deviation when the command is near-zero -- preserves standing quality
    # for the rel_standing_envs fraction without fighting a nonzero-command stride the rest of the time.
    stand_still_joints = RewTerm(
        func=loco_mdp.stand_still_joint_deviation_l1,
        weight=-0.2,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_pitch_joint", ".*_knee_joint", ".*_ankle_pitch_joint"]
            ),
        },
    )
    # (7) Feet-crossing/spreading guard -- LOOSENED live mid-session once real walking exposed that these
    # bounds were standing-task assumptions, not walking-task ones. Measured directly from the AMP reference
    # motion clip (policy_delay_walk_slow.npz, the actual demonstrated gait for this robot): left_y-right_y
    # separation ranges from -0.309 to +0.352 across the whole clip (mean 0.153, std 0.152) -- it's
    # constantly swinging through this entire range as NORMAL gait dynamics, not drifting toward instability.
    # min_separation loosened 0.1 -> -0.30 (just inside the real gait's own ~1st-percentile of -0.29) so this
    # only penalizes going narrower than real walking ever does, not the ordinary swing-through that was
    # getting punished the whole time.
    feet_separation = RewTerm(
        func=asimov1_mdp.feet_separation_l2,
        weight=-50.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
            "min_separation": -0.30,
        },
    )
    # feet_stance_width REMOVED live mid-session (was here in the standing task's reward shape, carried over
    # unexamined until the real-gait data above made the problem obvious): this term is a symmetric
    # attractor pulling continuously toward ONE fixed separation (0.215m, the robot's natural STANDING idle
    # width) -- correct for a task whose whole point is holding a static pose, actively wrong for a task
    # whose whole point is a continuously oscillating gait (real separation swings +-0.3m around a 0.153m
    # mean, per the reference clip). Keeping this term meant fighting the natural stride the entire time
    # it was engaged. The AMP discriminator already encodes the real gait's own oscillation pattern from the
    # reference clip directly, so removing this hand-shaped attractor loses no real guidance -- just the
    # part that was actively wrong for this task.
    # (8) Upper-body/waist damping -- unchanged from standing (arms/waist don't need to move for a basic walk).
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_yaw_joint"]
            )
        },
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.2, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_yaw_joint"])}
    )
    # (9) Shin/torso verticality. The original "cut weight 3x as a first guess" placeholder is now replaced
    # with measurements: checked both bodies directly against the AMP reference clip. torso tilt turns out
    # genuinely tiny during real walking (max ~0.13, barely more than standing) -- torso_vertical's weight
    # left as-is, it was never really the problem. Shin tilt is a different story: swing-phase shin tilt
    # reaches a 95th percentile of 0.77 and an absolute max of ~0.82 -- even at the reduced -0.3 weight, the
    # raw quadratic penalty was non-trivial exactly during a big, confident swing, fighting the exact stride
    # amplitude foot_clearance/foot_stuck/foot_swing_too_short were all trying to encourage. Given a real
    # dead-zone (max_tilt=0.85, just past the measured absolute max) instead of just a smaller blanket
    # weight -- same "measure the real extreme, tolerate up to it" discipline used for feet_crossed/
    # feet_too_wide, not another guess.
    shin_vertical = RewTerm(
        func=asimov1_mdp.body_vertical_l2,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_knee_link", "right_knee_link"]),
            "max_tilt": 0.85,
        },
    )
    # max_tilt dead-zone ADDED live mid-session per direct user question ("maybe we are being too strict on
    # his torso being vertical and when he leans forward to learn to walk he gets punished when he
    # shouldn't be?") -- real forward walking genuinely benefits from a deliberate forward lean for
    # propulsion (basic bipedal gait mechanics), and this term previously had NO tolerance for that at all
    # (unlike shin_vertical, which already got a dead-zone earlier). The reference clip's own torso tilt
    # was small (~0.13 max), but that's one specific demonstrated gait, not a hard ceiling on what THIS
    # robot's own best gait might need -- 0.3 gives real room for a deliberate lean without permitting
    # unlimited flailing.
    torso_vertical = RewTerm(
        func=asimov1_mdp.body_vertical_l2,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["waist_yaw_link"]), "max_tilt": 0.3},
    )
    # (10) Directional fall-recovery hint -- unchanged from standing. Only engages past tilt_threshold=0.15
    # (~8.6 deg of tilt), well beyond a normal walking lean, so it should stay dormant during ordinary gait
    # and still fire on genuine stumbles. Watch for any sign it's getting confused with normal stride lean.
    recovery_step_bonus = RewTerm(
        func=asimov1_mdp.recovery_step_bonus,
        weight=2.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            ),
            "tilt_threshold": 0.15,
            "tilt_range": 0.35,
        },
    )
    # (11) Smoothness penalties. action_rate weight raised -0.01 -> -0.1 for the SECOND fresh restart,
    # matching Menlo/cyclotron's own reported value. CORRECTED for the THIRD fresh restart: direct
    # comparison against Unitree H1's AND G1's own real recipes found BOTH actually use action_rate_l2=
    # -0.005 -- 20x GENTLER than cyclotron's -0.1, not the same number at all. A forward stride needs
    # bigger/faster joint-velocity changes than a cautious shuffle; an over-strong smoothness penalty would
    # suppress exactly that dynamic motion while leaving small corrective backward/sideways adjustments
    # cheap -- another plausible contributor to the forward-specific deficit, on top of the command-range
    # and actuator-stiffness fixes above. Trusting the two REAL, validated, currently-running biped tasks
    # over the one reported-secondhand number here.
    # weight raised -0.005 -> -0.03 (6x) for the gait-quality tuning pass (omnidirectional curriculum steps
    # 1-4 validated, user's own call to tune smoothness rather than widen the range further). Menlo's own
    # published action_rate weight is -1.0, NOT copied directly here -- magnitude-checked UNSAFE: logged
    # Episode_Reward/action_rate at the old weight was -0.3385, i.e. raw unweighted magnitude ~67.7/step.
    # At weight=-1.0 that term alone would contribute ~-67.7/step, ~35x track_lin_vel_xy's own weighted
    # contribution (~1.9-2.3) -- would almost certainly collapse the policy toward near-zero motion to avoid
    # the penalty, undoing all omnidirectional progress. This confirms Menlo's raw action_rate_l2 is computed
    # on a different scale than ours (different decimation/action formulation/normalization -- same class of
    # unit mismatch already flagged for their action_scale formula elsewhere in this project). Instead,
    # CALIBRATED our own target: -0.03 makes action_rate's weighted contribution (~67.7*0.03β‰ˆ2.0) comparable
    # in magnitude to track_lin_vel_xy's own contribution -- a real, meaningful 6x strengthening, not
    # negligible, but not dominant enough to freeze the policy either.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.03)
    dof_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    # (12) NEW gait-quality terms, added per Menlo's own published reward-design docs for this exact robot
    # (their "narrow-stance stability" and "contact-force limits" sections) -- same tuning pass as above.
    # leg_pose_tolerance: asymmetric per-joint-type tolerances (Menlo's own stated values, not guessed) --
    # see mdp.py's leg_pose_tolerance_l1 docstring for the dead-zone reasoning. Weight -0.5 is OUR OWN
    # calibrated starting guess (Menlo's own doc lists this term's weight as "curriculum"-scaled in THEIR
    # training, no fixed number to copy) -- flagged as a placeholder pending a real run's logged magnitude,
    # same discipline as every other un-measured numeric default in this project.
    leg_pose_tolerance = RewTerm(
        func=asimov1_mdp.leg_pose_tolerance_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint",
                    ".*_knee_joint", ".*_ankle_pitch_joint", ".*_ankle_roll_joint",
                ],
            ),
            "tolerances": {
                "hip_pitch": 0.5, "hip_roll": 0.25, "hip_yaw": 0.2,
                "knee": 0.5, "ankle_pitch": 0.2, "ankle_roll": 0.12,
            },
        },
    )
    # feet_stumble / feet_contact_force_limit: Menlo's own exact weights, magnitude-checked SAFE before
    # applying (unlike action_rate) -- feet_stumble is a bounded 0-2 count (naturally small, see mdp.py
    # docstring), feet_contact_force_limit's weight (-5e-4) is itself tiny enough that even a large force
    # excess stays comparable in magnitude to other terms, not dominant.
    feet_stumble = RewTerm(
        func=asimov1_mdp.feet_stumble,
        weight=-1.25,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            ),
        },
    )
    feet_contact_force_limit = RewTerm(
        func=asimov1_mdp.feet_contact_force_limit,
        weight=-5.0e-4,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            ),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    pelvis_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["pelvis_link"]), "threshold": 1.0},
    )
    collapsed = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.3})
    orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})
    hip_pitch_limit = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"bounds": (-1.2, 1.2), "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch_joint"])},
    )
    hip_roll_limit = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"bounds": (-0.65, 0.65), "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint"])},
    )
    hip_yaw_limit = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"bounds": (-0.65, 0.65), "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint"])},
    )
    # foot_airborne INTENTIONALLY REMOVED (present in the standing task) -- "any sustained foot-lift is bad"
    # is correct for standing and actively wrong for walking, which needs a real swing phase. This is the
    # single termination removal flagged in the project memory's "STRATEGIC DIRECTION" section as required
    # before this task could work at all.
    # threshold LOOSENED live mid-session -0.02 -> -0.35 once real walking began: the standing task's -0.02
    # default was tripping on 15.3% of the AMP reference clip's OWN frames (a real demonstrated walking
    # gait for this robot) -- measured directly from policy_delay_walk_slow.npz, not guessed. -0.35 sits
    # just past the clip's own absolute minimum (-0.309), a ~4cm margin beyond anything real walking does.
    feet_crossed = DoneTerm(
        func=asimov1_mdp.feet_crossed,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
            "threshold": -0.35,
        },
    )
    # max_separation LOOSENED live mid-session 0.32 -> 0.40 for the identical reason/data as feet_crossed
    # above -- the reference clip's own absolute maximum separation is 0.352m, so 0.32 clipped the top of
    # real walking's own natural range. 0.40 gives a ~5cm margin past the real max.
    feet_too_wide = DoneTerm(
        func=asimov1_mdp.feet_too_wide,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
            "max_separation": 0.40,
        },
    )
    # Added live mid-session per direct user suggestion ("maybe if we terminate if they stay in one spot
    # too long will encourage vector motion?") after persistent "bouncing/wiggling in place" was observed
    # despite balance/foot-lift/velocity-tracking rewards all engaging reasonably well -- none of them
    # measure NET progress, so oscillating in place could satisfy instantaneous reward without ever really
    # getting anywhere. See mdp.py's InsufficientProgressTermination docstring for the full reasoning
    # (including why this is a clean, non-exploitable failure condition, unlike the feet-crossing saga).
    # min_speed_fraction LOOSENED 0.3 -> 0.1 live mid-session: the first attempt at 0.3 fired on 82-90% of
    # episodes across a FULL 1000-iteration run with no downward trend (not "still learning," genuinely
    # stuck), while discriminator_prediction got WORSE (-0.60 -> -0.78) and a brand-new hip_pitch_limit
    # termination appeared (plausibly desperate flailing from being cut off so often) -- signs this was
    # actively destabilizing training, not nudging it. Rolled back to the checkpoint from BEFORE this
    # termination existed (model_7799, the previous best: 82.7% time_out) rather than building on top of
    # 1000 iterations that got measurably worse. 0.1 asks for only 10% of ideal tracking -- a true backstop
    # against zero-effort policies, not a constant harsh pressure.
    insufficient_progress = DoneTerm(
        func=asimov1_mdp.InsufficientProgressTermination,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
            "window_time": 2.0,
            "min_speed_fraction": 0.1,
        },
    )


@configclass
class ASIMOV1VelocityEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the asimov-1 velocity-commanded locomotion environment."""

    # Scene/actions/events reused VERBATIM from the standing task -- no changes needed for the pivot.
    scene: MySceneCfg = MySceneCfg(num_envs=128, env_spacing=3.0, clone_in_fabric=True)
    observations: Asimov1VelocityObservationsCfg = Asimov1VelocityObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 2
        self.episode_length_s = 20.0
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physics = Asimov1PhysicsCfg()
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0
        # action scale RAISED 0.25 -> 0.5 live mid-session (velocity/AMP task only, standing task
        # untouched) after direct comparison against Unitree H1/G1's own actual working recipe
        # (locomotion/velocity/velocity_env_cfg.py's shared ActionsCfg uses scale=0.5) per explicit user
        # request to check how Unitree's own real humanoid walking task does it rather than keep guessing.
        # 0.25 was set very early in this project (during the original STANDING task bring-up) specifically
        # to stop RANDOM exploration noise on an untrained network from tripping joint limits -- that
        # reasoning no longer applies to a mature, well-trained policy with controlled, deliberate action
        # std (~0.4-0.5, not raw noise). Running at HALF Unitree's own validated per-step positional
        # authority this whole time plausibly explains exactly the observed pattern: smaller corrective
        # backward/sideways catch-steps fit within the reduced range fine, while a full confident forward
        # stride's bigger coordinated joint excursion may not have had enough room to express itself.
        self.actions.joint_pos.scale = 0.5

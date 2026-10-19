# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Live policy-stitching demo: one simulated asimov-1 walks, gets pushed over, detects the fall,
recovers on its own, and resumes walking -- a continuous WALK -> FALL -> RECOVER -> WALK loop
chaining the walking policy (Isaac-Asimov1-Velocity-AMP-v0) and the fall-recovery policy
(Isaac-Asimov1-FallRecovery-v0) via an explicit sensor-gated state machine (see
asimov1_phase_stitch/gates.py for the exact thresholds and their sources).

Reuses the fall-recovery task's own env/scene as the single live simulation host (simpler of the
two tasks' scenes -- no command manager, no AMP obs group), UNWRAPPED (no RslRlVecEnvWrapper) so
this script has direct access to env.scene/env.observation_manager/raw env.step(). Both policies
are loaded straight from their actor.safetensors via asimov1_phase_stitch/policy_loader.py -- no
RSL-RL OnPolicyRunner needed for pure inference. See asimov1_phase_stitch/observations.py for why
the walking policy's 81-dim observation has to be assembled by hand each step (it was trained
against its own task's env, which isn't the one being simulated here), while the recovery policy's
76-dim observation comes directly from the reused env's own ObservationManager.

cfg.episode_length_s is set huge so the reused env's own time_out never force-resets it out from
under this script's own state machine -- every phase transition here is driven by the gates below,
not by the underlying task's own termination logic.

Exposes a slow-cadence status/control HTTP server (asimov1_phase_stitch/status_server.py) on
--status_port for external monitoring/triggering (e.g. h1v3-runtime's robot_supervisor agent) --
entirely separate from the 60Hz main loop, never blocking it.

AppLauncher-first construction order (same import-order-crash risk every script in this project
solves the same way: isaaclab_tasks/gymnasium must be imported only after AppLauncher constructs).

Usage (live-viewed, this project's standard GPU-safe recipe):
    __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia __VK_LAYER_NV_optimus=NVIDIA_only \\
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_asimov1_phase_stitch.py \\
        --walk_checkpoint /path/to/asimov1_omni_quality_run2_validated/actor.safetensors \\
        --recovery_checkpoint /path/to/asimov1_fallrecovery_validated/actor.safetensors \\
        --viz kit --kit_args="--/renderer/multiGpu/enabled=False --/renderer/activeGpu=0"
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Live walk->fall->recover->walk policy-stitching demo for asimov-1.")
parser.add_argument(
    "--walk_checkpoint", type=str, required=True, help="Path to the walking policy's actor.safetensors."
)
parser.add_argument(
    "--recovery_checkpoint", type=str, required=True, help="Path to the recovery policy's actor.safetensors."
)
parser.add_argument("--status_port", type=int, default=8800, help="Port for the status/control HTTP server.")
parser.add_argument(
    "--auto_push_interval_s",
    type=float,
    default=20.0,
    help="Seconds between automatic test pushes while walking (0 disables).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# everything below needs the simulation app already running -- this ordering is the actual fix.
import torch

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.classic.asimov1.asimov1_fall_recovery_env_cfg import ASIMOV1FallRecoveryEnvCfg

from asimov1_phase_stitch import gates, observations, policy_loader, push
from asimov1_phase_stitch.status_server import PhaseState, start_status_server

# NOT the brief's/OpenHorizon's own 0.25 -- that's a DIFFERENT policy's value. Confirmed directly
# from asimov1_velocity_env_cfg.py's own __post_init__: `self.actions.joint_pos.scale = 0.5`, a
# deliberate mid-project override (documented there: raised from 0.25 to match Unitree H1/G1's own
# convention) that ASIMOV1VelocityAmpEnvCfg inherits via super().__post_init__(). Using the wrong
# (smaller) scale here under-corrected every commanded joint target toward the walking default,
# which looked fine for about a second before destabilizing -- confirmed via a dedicated isolated
# diagnostic script before settling on this as the real root cause (several other candidates -- an
# action/observation rebase bug, PD gain mismatches, a blend-timing issue -- were ruled out first
# by direct instrumentation, not assumed away).
WALK_ACTION_SCALE = 0.5
RECOVERY_ACTION_SCALE = 0.5
WALK_DEMO_COMMAND = [0.4, 0.0, 0.0]  # fixed forward-walk command for the demo

# Action blending at the RECOVER->WALK handoff -- added after live testing confirmed the exact
# "seam" risk the brief flagged: the walking policy has never seen a post-recovery starting state
# (joint velocities, exact pose, facing direction all differ from its own training-time reset
# distribution), and an instantaneous switch (last_action jumping straight to zero, i.e. an
# immediate target of exactly default_joint_pos) was visibly destabilizing it -- observed live:
# every RECOVER->WALK handoff was followed by a fall-trigger again within ~1s, driven by angular
# speed spiking well past FALL_ANGVEL_RAD_S. Blending the commanded action from the last RECOVER
# target toward the walking policy's own fresh output over BLEND_TICKS ticks (instead of switching
# instantaneously) smooths the torque-target transition -- the brief's own recommended fallback
# once a non-blended handoff shows spikes, not a speculative addition.
# BLEND_TICKS dropped 20 -> 1 (effectively instant switch, no meaningful blend) after live testing
# with the corrected action scale (see WALK_ACTION_SCALE's own comment): standalone walking (no
# blend, full walking-policy control from tick 0) was rock-solid for 4.57s+ in an isolated
# diagnostic, while post-recovery walking WITH a 20-tick blend collapsed almost every time. Working
# theory: averaging the recovery policy's last target (a standing pose) with the walking policy's
# fresh target (a dynamic gait pose) produces a mushy intermediate joint-space target that's neither
# a good stand nor a good gait -- the blend was judged "not obviously helping" back when the wrong
# scale was still masking everything else, and was never re-evaluated after that fix landed. If this
# makes handoffs reliably WORSE instead of better, that's real evidence the blend wasn't the issue;
# raise this back up rather than assume.
BLEND_TICKS = 1

# Queued from last session, tried for the first time now (both cheap, both previously untried):
#
# (1) Zero residual joint velocity at the RECOVER->WALK handoff. `gates.standing_reached`'s
# STAND_MAX_ANG_SPEED check only looks at the robot's ROOT body angular velocity -- individual
# joint velocities (e.g. a leg still swinging from the recovery policy's own last tick) were never
# checked or cleared. The walking policy's very first post-handoff step could land on a root state
# that passes the standing gate while still riding on nonzero joint-space momentum the walking
# policy never saw at any point in its own training (which always starts from exactly zero joint
# velocity). Cheap and reversible: clear it explicitly at the handoff instant instead of trusting
# the existing gate to have implicitly covered it.
ZERO_JOINT_VEL_AT_HANDOFF = True

# (2) Ramp the walking command back up instead of immediately re-commanding the same confident
# WALK_DEMO_COMMAND the robot was cruising at before it got pushed over. A linear ramp from
# WALK_REENTRY_INITIAL_COMMAND up to WALK_DEMO_COMMAND over WALK_REENTRY_RAMP_S seconds of sim time,
# starting the instant WALK is (re-)entered -- applies uniformly to every WALK entry that sets
# walk_reentry_t (post-recovery handoff and a manual /control/reset both already set it; the very
# first WALK phase at script startup does not, and intentionally skips the ramp -- see
# _ramped_command below).
WALK_REENTRY_INITIAL_COMMAND = [0.15, 0.0, 0.0]
WALK_REENTRY_RAMP_S = 2.0


def _ramped_command(sim_time: float, reentry_t: float | None, device: str) -> torch.Tensor:
    if reentry_t is None:
        return torch.tensor([WALK_DEMO_COMMAND], device=device)
    frac = min(max((sim_time - reentry_t) / WALK_REENTRY_RAMP_S, 0.0), 1.0)
    cmd = [start + frac * (end - start) for start, end in zip(WALK_REENTRY_INITIAL_COMMAND, WALK_DEMO_COMMAND)]
    return torch.tensor([cmd], device=device)


def main():
    cfg = ASIMOV1FallRecoveryEnvCfg()
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 1.0e9  # never let this env's own time_out fire -- we own all resets

    env = gym.make("Isaac-Asimov1-FallRecovery-v0", cfg=cfg).unwrapped
    device = str(env.device)

    # The gravity-lift assist (apply_assist_force/AssistCurriculum, mdp.py) is a TRAINING-TIME env
    # mechanism, not a property of the checkpoint -- reusing this task's own env for the phase-
    # stitch scene means AssistCurriculum.__init__ sets the assist scale back to its initial value
    # (1.0, full assist) on every fresh env instance, and it only decays as the curriculum sees
    # resets happen. This script manages phase transitions manually and never calls env.reset() in
    # its normal cycle, so that curriculum never runs -- the assist force stays pinned at full
    # strength for the entire demo regardless of which checkpoint is loaded. Force it off explicitly
    # so the demo reflects the policy's actual fully-unassisted, validated (89% success) behavior.
    setattr(env, "asimov1_assist_scale", 0.0)

    print(f"[phase_stitch] loading walking policy from {args_cli.walk_checkpoint}")
    walk_policy = policy_loader.load_policy(args_cli.walk_checkpoint, action_scale=WALK_ACTION_SCALE, device=device)
    print(f"[phase_stitch] loading recovery policy from {args_cli.recovery_checkpoint}")
    recover_policy = policy_loader.load_policy(
        args_cli.recovery_checkpoint, action_scale=RECOVERY_ACTION_SCALE, device=device
    )

    # the reused env's own action term is hardcoded to the fall-recovery task's own values
    # (default_joint_pos=all-zero, scale=0.5) -- the recovery policy was trained against exactly
    # those, so its raw output can be passed to env.step() unmodified. The walking policy was
    # trained against a DIFFERENT default pose (a bent-knee stance, see observations.py's
    # WALKING_DEFAULT_JOINT_POS) and a different scale (0.25) -- `rebase_walk_action` converts its
    # raw output into whatever needs to be passed so env.step()'s own formula resolves to the
    # walking policy's true intended target. See observations.py's module docstring for the real
    # bug this fixes: using a naive scale-only ratio here (an earlier version of this script did
    # exactly that) ignores the default-pose mismatch entirely and silently sends systematically
    # wrong targets -- confirmed to destabilize the robot within ~1s even from a clean start.

    state = PhaseState()
    start_status_server(state, args_cli.status_port)

    env.reset()
    phase = "WALK"
    last_action_walk = torch.zeros(1, 23, device=device)
    last_action_recover = torch.zeros(1, 23, device=device)
    settle_timer = 0.0
    standing_hold_ticks = 0
    time_since_push = 0.0
    walk_reentry_t = None
    sim_time = 0.0
    handoff_action = torch.zeros(1, 23, device=device)  # last SCALED action applied before a WALK handoff
    blend_tick = BLEND_TICKS  # >= BLEND_TICKS means "not currently blending"

    print("[phase_stitch] ready, starting WALK phase.")
    try:
        with torch.inference_mode():
            while simulation_app.is_running():
                do_push, do_reset = state.drain_requests()
                if do_reset:
                    env.reset()
                    phase = "WALK"
                    last_action_walk = torch.zeros(1, 23, device=device)
                    walk_reentry_t = sim_time
                    blend_tick = BLEND_TICKS  # fresh env.reset() needs no blend, already a clean pose
                if do_push and phase == "WALK":
                    push.apply_push(env)

                if phase == "WALK":
                    if args_cli.auto_push_interval_s > 0:
                        time_since_push += env.step_dt
                        if time_since_push >= args_cli.auto_push_interval_s:
                            push.apply_push(env)
                            time_since_push = 0.0

                    command_buf = _ramped_command(sim_time, walk_reentry_t, device)
                    obs = observations.build_walk_obs(env, command_buf, last_action_walk)
                    raw_action = walk_policy.act(obs)
                    last_action_walk = raw_action
                    scaled_action = observations.rebase_walk_action(
                        env, raw_action, WALK_ACTION_SCALE, RECOVERY_ACTION_SCALE
                    )
                    if blend_tick < BLEND_TICKS:
                        weight = (blend_tick + 1) / BLEND_TICKS
                        scaled_action = weight * scaled_action + (1.0 - weight) * handoff_action
                        blend_tick += 1
                    env.step(scaled_action)

                    triggered, vals = gates.fall_triggered(env)
                    if triggered:
                        phase = "FALL"
                        settle_timer = 0.0
                        state.fall_count += 1
                    state.update(phase, vals, sim_time)

                elif phase == "FALL":
                    # hold the last commanded joint targets -- no new policy inference, just let
                    # physics/momentum/the PD controllers resolve the tumble.
                    held_action = observations.rebase_walk_action(
                        env, last_action_walk, WALK_ACTION_SCALE, RECOVERY_ACTION_SCALE
                    )
                    env.step(held_action)
                    settle_timer += env.step_dt

                    settled, vals = gates.settle_reached(env)
                    if settled or settle_timer >= gates.SETTLE_TIMEOUT_S:
                        phase = "RECOVER"
                        last_action_recover = torch.zeros(1, 23, device=device)
                        standing_hold_ticks = 0
                    state.update(phase, vals, sim_time)

                elif phase == "RECOVER":
                    # no rebasing needed: the recovery policy was trained against exactly the
                    # reused env's own default_joint_pos/scale, so its raw output is already what
                    # env.step() expects.
                    obs = observations.get_recover_obs(env)
                    raw_action = recover_policy.act(obs)
                    last_action_recover = raw_action
                    scaled_action = raw_action
                    env.step(scaled_action)

                    standing, vals = gates.standing_reached(env)
                    standing_hold_ticks = standing_hold_ticks + 1 if standing else 0
                    if standing_hold_ticks >= gates.STAND_HOLD_TICKS:
                        if ZERO_JOINT_VEL_AT_HANDOFF:
                            asset = env.scene["robot"]
                            asset.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(asset.data.joint_vel.torch))
                        phase = "WALK"
                        last_action_walk = torch.zeros(1, 23, device=device)
                        time_since_push = 0.0
                        walk_reentry_t = sim_time
                        handoff_action = scaled_action
                        blend_tick = 0
                    state.update(phase, vals, sim_time)

                if walk_reentry_t is not None and phase == "WALK":
                    if sim_time - walk_reentry_t >= gates.WALK_REFALL_WINDOW_S:
                        state.walk_success_count += 1
                        walk_reentry_t = None
                        print(f"[phase_stitch] t={sim_time:.1f}s recovery cycle CONFIRMED (4s no re-fall)")

                sim_time += env.step_dt
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

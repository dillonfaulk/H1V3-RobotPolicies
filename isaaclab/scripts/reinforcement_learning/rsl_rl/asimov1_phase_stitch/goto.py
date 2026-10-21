"""v0 go-to-point FSM: orient-then-move proportional controller riding on the trained omni policy's
own velocity-command interface. This is architecture (a) from arXiv:2508.14098 ("No More Marching:
Learning Humanoid Locomotion for Short-Range SE(2) Targets") -- needs zero new training and zero new
checkpoint, and rides entirely on the already-robustness-validated walking checkpoint
(asimov1_omni_robustness_validated). Chosen over an end-to-end trained goal policy (v1, not built
yet) for the reason that actually matters in a critical setting: every command this emits traces to
two numbers (distance, heading_error) and this file, its failure mode is bounded by the clamps below
(worst case: a slow robot that stops), and it needs no new checkpoint whose edge cases would
otherwise be discovered live. See asimov1-goto-point-task-brief.md, sections 2 and the muse addendum,
for the full design discussion -- the planned v1 (trained goal-conditioned policy) is meant to run
INSIDE this same supervisor later, not replace it.

Command-magnitude bounds are not arbitrary -- they're read directly from the currently-deployed
walking checkpoint's own trained command envelope (asimov1_omni_robustness_validated/metadata.json's
command_ranges), confirmed by file read, not assumed from the task's current (possibly since-widened)
env cfg. Commanding outside a checkpoint's trained range is exactly the kind of out-of-distribution
risk this whole v0-over-v1 argument is supposed to avoid -- so these bounds get the same sourcing
discipline as gates.py's own thresholds.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_apply_inverse

# Trained command envelope for asimov1_omni_robustness_validated (confirmed via its own metadata.json
# command_ranges, not the task cfg's current/possibly-wider ranges -- this script always loads that
# specific checkpoint's weights, so its OWN training distribution is the one that matters here).
LIN_VEL_X_RANGE = (-0.7, 0.8)
ANG_VEL_Z_RANGE = (-0.8, 0.8)

ARRIVAL_RADIUS_M = 0.25  # brief's v0 acceptance bar
ARRIVAL_HOLD_TICKS = 15  # ~0.25s at 60Hz -- require a sustained arrival, not a momentary pass-through,
# before advancing to the next waypoint. Same consecutive-tick discipline already used twice elsewhere
# in this project's reward shaping (StabilityHoldBonus, RecoveryFeetPlantedBonus) for the same reason:
# a momentary condition is cheap to satisfy by accident, a sustained one isn't. Also gives the FSM's
# own command a real "stop and hold" moment at each waypoint instead of an instant replan mid-stride.

# Orient-then-move as a smooth taper, not a hard v0/v1 switch -- revised after the first live test
# (2026-10-20) showed the original hard ORIENT_THRESHOLD_RAD cutoff (turn-in-place below it, move
# above it) repeatedly triggered spin-type falls (ang_speed > FALL_ANGVEL_RAD_S while the robot
# stayed upright, i.e. NOT a balance/tilt failure -- a whipsaw failure). Root cause: heading_error
# flips sign as the robot oscillates around alignment, and each flip was commanding a full-magnitude
# turn-rate reversal with no rate limiting -- exactly the kind of out-of-distribution command
# combination (max |ang_vel_z| sustained, zero lin_vel_x) the brief itself warned v0 needs to stay
# inside the checkpoint's trained envelope for. Two independent fixes below address this directly:
# a continuous forward-speed taper (no discontinuity at a threshold) and an explicit slew-rate limit
# on the command itself (so even a genuine heading_error sign flip can't instantly reverse the
# commanded turn rate).
HEADING_FULL_SPEED_RAD = 0.3  # below this: full taper-computed forward speed
HEADING_STOP_RAD = 1.0  # at/above this: zero forward speed, pure pivot-turn
TAPER_MIN = 0.0  # tried 0.15 in a second live test (2026-10-20) to avoid a true v=0 pivot-turn, on the
# theory that a stationary spin was itself destabilizing the gait -- it WAS safer against spin-falls, but
# live-observed side effect was worse: a small constant forward crawl sustained through the many seconds a
# sharp reorient takes adds up to real distance in the wrong direction, so the robot swung past the target
# in a wide arc instead of turning tightly to it (user, watching live: "he went past it and cant turn
# enough as hes walking... taking a huge wide radius turn"). Reverted to a true stop-and-pivot for large
# heading error -- zero forward drift while turning is worth more than avoiding a stationary spin, given
# the first live test (taper-to-zero, CAP=0.5) already cleared 2/2 full waypoint legs with only the
# hardest case (a near-180 deg reorient) causing trouble. Splitting the difference on CAP below instead.

K_V = 1.0  # command_v = clip(K_V * distance, 0, V_MAX_MS) * taper(heading_error)
V_MAX_MS = 0.5  # brief's starting point; comfortably inside LIN_VEL_X_RANGE's own 0.8 ceiling
K_W = 1.0  # command_w = clip(K_W * heading_error, -ANG_VEL_CMD_CAP, ANG_VEL_CMD_CAP)
ANG_VEL_CMD_CAP = 0.4  # splits the difference between the first live test's 0.5 (worked for 2/2 easier
# legs, destabilized only on the hardest ~180 deg reorient) and the second's overcorrected 0.3 (safe but
# the TAPER_MIN=0.15 combination above is what actually caused the circling, not this cap specifically --
# kept a little lower than 0.5 anyway for some real margin on sustained sharp turns). v0's own self-imposed
# ceiling either way, well inside the checkpoint's full trained ANG_VEL_Z_RANGE (+-0.8) -- see K_V's own
# comment on why that margin matters for a bounded-failure-mode controller.

MAX_DV_PER_TICK = 0.03  # 0 -> V_MAX_MS ramps over ~17 ticks (~0.28s at 60Hz)
MAX_DW_PER_TICK = 0.04  # a full -CAP -> +CAP reversal takes ~20 ticks (~0.33s) -- no more instant flips

# Visual markers so the goal(s) are actually visible in the Kit viewport (added after live testing
# showed the goto behavior with no way to see the target point in the scene). One prototype per
# marker: "current" (bright green, larger -- the goal actively being steered toward) and "queued"
# (orange, smaller -- waypoints still waiting in the queue). Mirrors Isaac Lab's own
# POSITION_GOAL_MARKER_CFG pattern (isaaclab.markers.config) rather than inventing a new one.
WAYPOINT_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/GotoWaypoints",
    markers={
        "current": sim_utils.SphereCfg(
            radius=0.1, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
        ),
        "queued": sim_utils.SphereCfg(
            radius=0.07, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.6, 0.0))
        ),
    },
)
WAYPOINT_MARKER_HEIGHT_M = 0.4  # chest-ish height on this robot -- easy to spot against the floor grid


def make_waypoint_visualizer() -> VisualizationMarkers:
    """Must be called after the simulation app is running (same ordering constraint as every other
    sim-stage-touching call in this project) -- so this is a factory, not a module-level singleton."""
    return VisualizationMarkers(WAYPOINT_MARKER_CFG)


def update_waypoint_markers(
    visualizer: VisualizationMarkers,
    current_goal: tuple[float, float] | None,
    queued_goals: list[tuple[float, float]],
    device: str,
) -> None:
    all_goals = ([current_goal] if current_goal is not None else []) + list(queued_goals)
    if not all_goals:
        visualizer.set_visibility(False)
        return
    visualizer.set_visibility(True)
    translations = torch.tensor(
        [[g[0], g[1], WAYPOINT_MARKER_HEIGHT_M] for g in all_goals], device=device
    )
    marker_indices = [0] + [1] * (len(all_goals) - 1) if current_goal is not None else [1] * len(all_goals)
    visualizer.visualize(translations=translations, marker_indices=marker_indices)


def compute_goto_command(
    env, goal_xy: tuple[float, float], device: str, prev_command: torch.Tensor | None = None
) -> tuple[torch.Tensor, dict]:
    """World-frame (x, y) goal -> a [lin_vel_x, lin_vel_y, ang_vel_z] command in the robot's own
    base frame, via the same global->local transform pattern confirmed in catachiii/crl-humanoid-ros's
    real G1 goal controller (quat_apply_inverse on the root quaternion, not a hand-rolled yaw
    extraction -- fewer places for a sign convention to go wrong, and this project has been bitten by
    exactly that class of bug before, e.g. the old FORWARD_VEC_B lesson).

    `prev_command` (this function's own previous return value, threaded by the caller the same way
    last_action_walk/last_action_recover already are in play_asimov1_phase_stitch.py) enables the
    slew-rate limit above -- pass None only on the very first call for a freshly-activated goal.

    Returns (command_tensor[1,3], vals) -- vals is a plain-float dict (distance, heading_error,
    arrived, goal_x, goal_y) for status-server/console reporting, mirroring gates.py's own
    (bool, raw_values_dict) convention.
    """
    asset = env.scene["robot"]
    root_pos = asset.data.root_pos_w.torch[0]
    root_quat = asset.data.root_quat_w.torch[0]  # confirmed (x, y, z, w) via base_articulation_data.py's
    # own QUAT_XYZW_ELEMENT_NAMES annotation on root_quat_w in this pinned Isaac Lab version -- matches
    # isaaclab.utils.math's own documented (x, y, z, w) convention, so quat_apply_inverse needs no
    # reordering here.

    goal_world = torch.tensor([goal_xy[0], goal_xy[1], root_pos[2].item()], device=device)
    goal_vec_world = (goal_world - root_pos).unsqueeze(0)
    goal_vec_local = quat_apply_inverse(root_quat.unsqueeze(0), goal_vec_world)[0]
    x_local, y_local = goal_vec_local[0], goal_vec_local[1]

    distance = torch.sqrt(x_local * x_local + y_local * y_local)
    heading_error = torch.atan2(y_local, x_local)
    abs_heading = abs(heading_error.item())

    arrived = bool((distance < ARRIVAL_RADIUS_M).item())
    if arrived:
        command = [0.0, 0.0, 0.0]
    else:
        raw_taper = (HEADING_STOP_RAD - abs_heading) / (HEADING_STOP_RAD - HEADING_FULL_SPEED_RAD)
        raw_taper = max(min(raw_taper, 1.0), 0.0)
        taper = TAPER_MIN + (1.0 - TAPER_MIN) * raw_taper
        v = max(min(K_V * distance.item(), V_MAX_MS), 0.0) * taper
        v = max(min(v, LIN_VEL_X_RANGE[1]), LIN_VEL_X_RANGE[0])
        w = max(min(K_W * heading_error.item(), ANG_VEL_CMD_CAP), -ANG_VEL_CMD_CAP)
        command = [v, 0.0, w]

    cmd_tensor = torch.tensor([command], device=device)
    if prev_command is not None:
        dv = torch.clamp(cmd_tensor[0, 0] - prev_command[0, 0], -MAX_DV_PER_TICK, MAX_DV_PER_TICK)
        dw = torch.clamp(cmd_tensor[0, 2] - prev_command[0, 2], -MAX_DW_PER_TICK, MAX_DW_PER_TICK)
        cmd_tensor = torch.tensor(
            [[prev_command[0, 0].item() + dv.item(), 0.0, prev_command[0, 2].item() + dw.item()]], device=device
        )

    vals = {
        "distance": round(distance.item(), 3),
        "heading_error": round(heading_error.item(), 3),
        "arrived": arrived,
        "goal_x": goal_xy[0],
        "goal_y": goal_xy[1],
    }
    return cmd_tensor, vals

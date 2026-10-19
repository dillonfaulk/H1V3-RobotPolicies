"""Phase-transition gates for the walk -> fall -> recover -> walk state machine.

Thresholds are all sourced and documented at the call site, not guessed:
- Fall trigger: OpenHorizon's own real source (`src/asimov_policies/common/fall_trigger.py`,
  openhorizon-labs/asimov-v1-policies), pulled directly, not a secondhand summary.
- Settle gate: OpenHorizon's own docs ("near-still: base speed < 0.2 m/s and < 0.5 rad/s"). A
  separate research brief claimed slightly different numbers (0.15 m/s, explicit 0.3s hold) from a
  source not yet independently verified against OpenHorizon's actual settle-gate code -- flagged as
  a fast-follow, not blocking. SETTLE_TIMEOUT_S is a safety net so the state machine never waits
  forever if the robot doesn't settle cleanly.
- Standing gate: 0.9*STANDING_HEIGHT / 0.9 upright is the exact formula asimov-1's own fall-recovery
  task already uses internally for its `AssistCurriculum` success gate (same numbers, independently
  corroborated against OpenHorizon's own getup standing-gate, not a new guess for this script).

Each function returns (bool, raw_values_dict) so the status server can report the exact numbers
that caused a transition, not just the boolean outcome.
"""

from __future__ import annotations

import torch

FALL_TILT_DEG = 30.0
FALL_ANGVEL_RAD_S = 2.0

SETTLE_LINVEL_MS = 0.2
SETTLE_ANGVEL_RAD_S = 0.5
SETTLE_TIMEOUT_S = 3.0

STANDING_HEIGHT = 0.63
STAND_MIN_PELVIS_HEIGHT = 0.9 * STANDING_HEIGHT  # 0.567, matches AssistCurriculum's own gate
STAND_MIN_UPRIGHT = 0.9  # on -projected_gravity_b[:, 2]
# STAND_MAX_ANG_SPEED added after live testing caught a real gap: height+upright alone says nothing
# about whether the robot has actually settled -- a robot mid-recovery can momentarily satisfy both
# while still spinning hard (observed live: ang_speed 2.7-4.5 rad/s at the moment of a height+
# upright-only "standing" gate firing), handing off to the walking policy while still rotating,
# which immediately re-triggers the fall detector via its own ang-speed condition. 1.0 gives a real
# margin below FALL_ANGVEL_RAD_S=2.0 without being strict.
STAND_MAX_ANG_SPEED = 1.0
# STAND_HOLD_TICKS: tiny 3-tick debounce, confirmed the right choice TWICE now. OpenHorizon's own
# 2.0s hold and this policy's old `StabilityHoldBonus` 25-step hold both got the robot stuck in
# RECOVER indefinitely early in this project. TRIED AGAIN at 20 ticks later in this same session
# (after two reward-shape attempts failed to fix the knee-bend-at-handoff value, as a cheap test of
# whether a longer settle time would help) -- got stuck again just as badly (106+s in RECOVER,
# ang_speed oscillating well above STAND_MAX_ANG_SPEED the whole time, confirmed live by the user:
# "he was already waiting a little too long before this change and now its worse"). Even with
# `recovery_feet_planted`/`stability_hold` now training real continuous-hold behavior (up to
# 30/25 consecutive ticks respectively), the policy apparently still can't reliably sustain
# ang_speed<1.0 for 20 CONSECUTIVE ticks in a row -- whatever residual motion it has at the terminal
# pose is bursty/intermittent, not a smooth settle, so a longer consecutive-tick requirement just
# means waiting for an increasingly unlikely lucky streak rather than genuinely more settle time.
# Reverted back to 3 -- this knob is not the fix for the knee-bend timing question; that remains
# open for a future session (e.g. a dedicated "sustained low ang_speed" reward term might succeed
# where a phase-machine gate change didn't).
STAND_HOLD_TICKS = 3
# feet_in_contact was also part of AssistCurriculum's real 3-condition formula and had been missed
# here entirely -- added below via a sensor_cfg param, matching the exact asset limitation already
# documented elsewhere in this project (only the 2 ankle_roll bodies + pelvis have PhysX contact-
# report coverage on this asset).
STAND_CONTACT_THRESHOLD_N = 1.0

WALK_REFALL_WINDOW_S = 4.0  # success criterion: no re-fall for this long after re-entering WALK

_FALL_COS_TILT = torch.cos(torch.deg2rad(torch.tensor(FALL_TILT_DEG))).item()


def fall_triggered(env) -> tuple[bool, dict]:
    """Tilt from vertical exceeds FALL_TILT_DEG, OR angular speed exceeds FALL_ANGVEL_RAD_S."""
    asset = env.scene["robot"]
    gravity_b = asset.data.projected_gravity_b.torch[0]
    ang_vel = asset.data.root_ang_vel_w.torch[0]

    tilted = bool(-gravity_b[2].item() < _FALL_COS_TILT)
    ang_speed = float(torch.norm(ang_vel).item())
    spinning = ang_speed > FALL_ANGVEL_RAD_S

    values = {"neg_gravity_z": -gravity_b[2].item(), "cos_tilt_threshold": _FALL_COS_TILT, "ang_speed": ang_speed}
    return (tilted or spinning), values


def settle_reached(env) -> tuple[bool, dict]:
    """Near-still: linear speed < SETTLE_LINVEL_MS AND angular speed < SETTLE_ANGVEL_RAD_S."""
    asset = env.scene["robot"]
    lin_vel = asset.data.root_lin_vel_w.torch[0]
    ang_vel = asset.data.root_ang_vel_w.torch[0]

    lin_speed = float(torch.norm(lin_vel).item())
    ang_speed = float(torch.norm(ang_vel).item())
    settled = lin_speed < SETTLE_LINVEL_MS and ang_speed < SETTLE_ANGVEL_RAD_S

    return settled, {"lin_speed": lin_speed, "ang_speed": ang_speed}


_FEET_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]


def standing_reached(env) -> tuple[bool, dict]:
    """Pelvis above STAND_MIN_PELVIS_HEIGHT AND upright past STAND_MIN_UPRIGHT AND both feet in
    contact AND not still spinning. The first three conditions exactly match `AssistCurriculum`'s
    own real success formula (confirmed by direct re-read of that class, not memory) -- the formula
    that actually produced this checkpoint's measured 89% success rate. ang_speed is the one
    addition beyond that formula (see STAND_MAX_ANG_SPEED's own comment)."""
    asset = env.scene["robot"]
    height = float(asset.data.root_pos_w.torch[0, 2].item())
    upright = float(-asset.data.projected_gravity_b.torch[0, 2].item())
    ang_speed = float(torch.norm(asset.data.root_ang_vel_w.torch[0]).item())

    contact_sensor = env.scene.sensors["contact_forces"]
    feet_ids, _ = contact_sensor.find_bodies(_FEET_BODY_NAMES)
    feet_forces = contact_sensor.data.net_forces_w.torch[0, feet_ids, :]
    feet_in_contact = bool((torch.norm(feet_forces, dim=-1) > STAND_CONTACT_THRESHOLD_N).all().item())

    standing = (
        height > STAND_MIN_PELVIS_HEIGHT
        and upright > STAND_MIN_UPRIGHT
        and feet_in_contact
        and ang_speed < STAND_MAX_ANG_SPEED
    )
    return standing, {"height": height, "upright": upright, "feet_in_contact": feet_in_contact, "ang_speed": ang_speed}

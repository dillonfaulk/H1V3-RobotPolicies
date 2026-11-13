"""BallKick mission: go to the ball, line up, swap in the kick policy, kick, swap back.

Run from the microduck_rl repo root:
  uv run python scripts/kick_mission.py <velstand.onnx> <kick_right.onnx> [--runs 5]
  uv run python scripts/kick_mission.py <velstand.onnx> <kick_right.onnx> --viewer [--ball 1.2,0.3 --kick-heading 30]

Phases (one control loop at 50 Hz, VelStand driving except during KICK):
  NAVIGATE  go-to-point supervisor (scripts/goto_supervisor.py, unchanged) to a
            point on the kick line behind the stance spot, via a side detour if
            the straight path would cross the ball.
  ALIGN     pulse-and-settle fine alignment until the ball sits at the kick
            policy's trained offset in the duck's yaw frame (0.09, -0.042)
            within +-ALIGN_TOL and the heading is within ALIGN_HEADING_TOL of
            the kick direction. Measured round-3 pulse responses (from rest,
            settled): pivot 0.8 rad/s ~0.8 deg/tick with ~no translation;
            strafe 0.15 m/s x5 ticks ~ +23 mm left / -11 mm right; forward
            0.2 m/s <= 10 ticks shuffles 3-4 mm, 15 ticks is one ~38 mm step;
            backward barely moves (-7 mm at 20 ticks).
  KICK      hard ONNX swap to the kick policy (same 61-D buffer, all-zero
            command as in kick training), held KICK_HOLD_S, then swap back.
  SETTLE    VelStand, zero twist. A fall here gets the go-to-point fall rule:
            zero twist until upright 1 s.
With --precision <onnx> the final approach is the PrecisionStep policy instead:
  PRECISE_PIVOT  VelStand pivots onto the kick line (|heading err| < 8 deg; the
                 step policy trained on +-15 deg).
  PRECISE        hard swap to the step policy; the 3 twist slots carry the
                 remaining offset to the stance pose [dx, dy, dyaw] in the trunk
                 yaw frame (option A). On target (+-12 mm, 3 deg) and still for
                 0.5 s -> KICK directly.
  The pulse ALIGN stays as the FALLBACK: a 6 s timeout or a fall in PRECISE swaps
  VelStand back in and hands over to it; a bumped ball re-enters PRECISE at most
  twice. Every mission records its approach_mode.
ALIGN uses its own clamps (|vx| <= 0.2, |vy| <= 0.15, pulses <= 20 ticks, only
within ALIGN_ZONE_M of the stance): go-to-point's forward-only / no-strafe
clamps can't place the duck to +-15 mm.
"""
import argparse
import math
import os
import sys
import time

import mujoco
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import infer_policy as ip  # noqa: E402
from goto_point import CONTROL_DT, RunLog  # noqa: E402
from goto_supervisor import GotoConfig, GotoSupervisor, wrap_angle  # noqa: E402
from kick_battery import BALL_OFFSET_ABS_Y, BALL_OFFSET_X, SCENES, KickSim  # noqa: E402

ARENA = (-2.5, 2.5, -2.5, 2.5)
PRE_STANCE_M = 0.12        # approach point this far behind the stance spot, on the kick line
LINE_UP_M = 0.40           # a second waypoint further back, so the last leg runs down the kick line
DETOUR_CLEAR_M = 0.25      # straight path closer than this to the ball -> side detour
ALIGN_ZONE_M = 0.30        # ALIGN clamps only apply this close to the stance spot
ALIGN_TOL = 0.012          # m, lateral: inside the kick policy's +-15 mm training noise
ALIGN_X_WINDOW = (-0.020, 0.012)  # m, along the kick line (+ = ball farther than nominal), see Aligner.pulse
BALL_WAIT_S = 6.0          # max wait for a bumped ball to stop (the sim ball has ~no rolling resistance)
APPROACH_FROM = 0.06       # m of along-line error above which ALIGN walks in continuously
APPROACH_STOP = 0.012      # cut the walk here (trunk-relative); it lands ~0-20 mm further on
STEP_MIN = 0.045           # a 15-tick step's swing reaches the ball from closer than this (it kicked it 0.2-1.9 m)
MAX_BACKOFFS = 2          # then correct in place rather than loop
BACKOFF_TO = 0.08          # backing up (continuous, it works; short back pulses don't) until this far out
ALIGN_HEADING_TOL = math.radians(3)
ALIGN_SETTLE_TICKS = 40    # 0.8 s standing between pulses
ALIGN_MAX_PULSES = 40
SQUARE_TOL = float(os.environ.get("KICK_SQUARE_TOL", "0.015"))  # kicking-foot x vs HOME before a swap-in
SUPPORT_TOL = float(os.environ.get("KICK_SUPPORT_TOL", "0.012"))  # support-foot x vs HOME
SQUARE_MAX = 3
# Squaring move: strafe left 5 ticks then right 10 (settling between). Branched from the
# real staggered stances of failed missions, it brought the support foot to within
# 1.3 mm of HOME and the kicking foot to 7-8 mm in all three; back 20 only fixed a
# kicking foot that was behind, pivot pairs barely changed the stance, and 12-tick
# forward pulses knocked the ball. It shifts the duck ~3 cm sideways, which the
# lateral pulses then take back out.
SQUARE_MOVE = [((0.0, 0.15, 0.0), 5), ((0.0, -0.15, 0.0), 10)]
KICK_HOLD_S = 2.0
SETTLE_S = 2.0
# Precision-step final approach (--precision <onnx>; Mjlab-PrecisionStep-Flat-MicroDuck).
# The pulse aligner above stays as the FALLBACK: a precision timeout, a fall, or
# too many re-entries hands the last stretch back to it.
PRECISE_PIVOT_TOL = math.radians(8)   # VelStand pivots first: the step policy trained on +-15 deg
PRECISE_PIVOT_MAX_S = 5.0
PRECISE_STILL_V = 0.02                # m/s trunk speed counted as "stopped"
PRECISE_HOLD_TICKS = 25               # 0.5 s on target and still -> kick
PRECISE_TIMEOUT_S = 6.0               # = the training episode length
PRECISE_MAX_ENTRIES = 2               # (re-)entries after a ball bump, then pulses
PRECISE_MAX_R = 0.25                  # offset clipped to the training range
PRECISE_MAX_YAW = math.radians(15)
# Calibration (2026-11-11, model_3999, seeds 0+1, 60 missions): the step policy parks the
# trunk on target but the feet end ~5 mm behind HOME, so measured from the kicking foot the
# ball sat +10.2 mm (median; 0.3-14.2) farther than the kick trained on and rolled a median
# 1.01 m (the pulse aligner, which parks at +0.7 mm, rolls 1.22 m). The stance target handed
# to the step policy moves this far forward along the kick line. Precision path only: the
# pulse fallback keeps its own (foot-relative) target.
# Per kicking foot: each foot's kick checkpoint is its own measurement.
# LEFT (2026-11-12, BallKick left ts0.25 model_1250, seeds 0+1, 60 missions): 0 by decision.
# Unshifted, the ball sits +9.4 mm far of the kick spot but already rolls 1.19 m (60/60,
# 0 bumps). +9 mm was tested and REJECTED: 2/60 bumps, the step policy's stepping foot clipped
# the ball during PRECISE (PrecisionStep trained with the ball on the right side only).
PRECISE_STANCE_SHIFT_M = {"right": float(os.environ.get("KICK_STANCE_SHIFT", "0.010")),
                          "left": float(os.environ.get("KICK_STANCE_SHIFT_LEFT", "0.0"))}
# Aim offset for BallKick v1's (right foot, ts0.25 model_1000) left drift. The duck lines up
# facing kick_heading + KICK_AIM_OFFSET; the ball result is still measured against kick_heading.
# Sign: +yaw is LEFT (CCW, z up), kick_dir_err_deg > 0 means the ball went left, so the offset
# is NEGATIVE (aim right). Calibrated 2026-11-11 against the +10 mm stance above: drift
# +17.9 deg median (std 1.7, IQR 16.5..19.1, flat across ball distance), 60 missions, seeds 0+1.
# LEFT (BallKick left ts0.25 model_1250, 2026-11-12): drifts RIGHT, -11.2 deg median (std 1.3,
# IQR -12.3..-10.3, flat across ball distance) at the unshifted stance, so the offset is
# POSITIVE (aim left). 60 missions, seeds 0+1.
KICK_AIM_OFFSET = {"right": math.radians(float(os.environ.get("KICK_AIM_OFFSET_DEG", "-17.9"))),
                   "left": math.radians(float(os.environ.get("KICK_AIM_OFFSET_DEG_LEFT", "11.2")))}


def clamp_align(vx, vy, wz):
    return (min(max(vx, -0.2), 0.2), min(max(vy, -0.15), 0.15), min(max(wz, -1.0), 1.0))


class Aligner:
    """Chooses one pulse at a time from the settled ball-in-duck-frame error."""

    def __init__(self, foot="right"):
        self.target = np.array([BALL_OFFSET_X, -BALL_OFFSET_ABS_Y if foot == "right" else BALL_OFFSET_ABS_Y])
        self.last_axis, self.last_err, self.gain = None, None, 1.0

    def errors(self, pose, ball_xy, kick_heading, foot_local=None, home_foot=None):
        """Ball placement error in the duck's yaw frame (+ ex: ball too far ahead, + ey: too far left).

        With the kicking foot's position (foot_local, and where it sits at HOME), the
        error is measured foot-relative: after walking and pulses the feet can end up
        staggered by 2+ cm, and a trunk-relative fit then misses the kick (measured:
        a 23 mm foot stagger turned an in-tolerance trunk fit into a whiff).
        """
        x, y, yaw = pose
        d = np.asarray(ball_xy) - (x, y)
        c, s = math.cos(yaw), math.sin(yaw)
        local = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
        if foot_local is not None:
            ex, ey = (local - foot_local) - (self.target - home_foot)
        else:
            ex, ey = local - self.target
        return ex, ey, wrap_angle(yaw - kick_heading), local

    def _escalate(self, axis, err):
        """Pulse length multiplier: grows while the same axis keeps making < 30% progress.

        The measured pulse table is from a square stance; with the feet split a short
        pivot can do nothing at all (seen: 36 identical 4-tick pivots, heading stuck).
        """
        if axis == self.last_axis and abs(err) > 0.7 * abs(self.last_err):
            self.gain = min(self.gain * 1.6, 4.0)
        else:
            self.gain = 1.0
        self.last_axis, self.last_err = axis, err
        return self.gain

    def pulse(self, ex, ey, eh, allow_backoff=True):
        """(twist, ticks, axis) or None when aligned."""
        if allow_backoff and ex < 0.0 and (abs(ey) > 0.020 or abs(eh) > math.radians(15)):
            # Ball already closer than nominal and a big sideways/heading fix to make:
            # the strafe or long pivot sweeps the kicking foot through the ball (5 of 7
            # bumps in 60 missions). Back off and walk in again instead. (Heading only
            # past 15 deg: walk-ins routinely end 8-10 deg off, and backing off for
            # that looped forever.)
            return None, 0, "backoff"
        if abs(eh) > ALIGN_HEADING_TOL:
            g = self._escalate("heading", eh)
            k = int(np.clip(round(g * math.degrees(abs(eh)) / 0.8), 2, 20))
            return (0.0, 0.0, -math.copysign(0.8, eh)), k, "heading"
        if abs(ey) > ALIGN_TOL:
            g = self._escalate("lateral", ey)
            rate = 4.6 if ey > 0 else 2.1  # mm per tick (measured, <= 5 ticks)
            k = int(np.clip(round(g * abs(ey) * 1000 / rate), 2, 10))
            return (0.0, math.copysign(0.15, ey), 0.0), k, "lateral"
        self._escalate("x", ex)
        # Forward is quantized: <= 10 ticks only twists the duck in place, 15 ticks is
        # one step (25-35 mm in the loop, plus a few degrees of yaw the heading pulse
        # then fixes). The x window is lopsided because the kick is (measured,
        # ts0.25 model_1000): a ball up to 30 mm CLOSER than nominal is always kicked,
        # farther than +15 mm it starts to whiff (+20: 7/8, +25: 5/8, +40: 0/8). A
        # step from just past the far edge lands inside, on the close side.
        if ex > STEP_MIN:
            return (0.2, 0.0, 0.0), 15, "step"
        if ex > ALIGN_X_WINDOW[1]:
            return None, 0, "backoff"   # too close to step safely, too far to kick: back off, walk in again
        if ex < ALIGN_X_WINDOW[0]:
            return (-0.2, 0.0, 0.0), 20, "back"     # ~7 mm
        return None


class Mission:
    def __init__(self, sim, kick_session, kick_heading, log, foot="right", precision_session=None):
        self.sim, self.kick_session = sim, kick_session
        self.kh_goal = kick_heading                              # where the ball should go
        self.foot = foot
        self.kh = wrap_angle(kick_heading + KICK_AIM_OFFSET[foot])  # where the duck faces (aim-corrected)
        self.precision_session = precision_session
        self.precise_entries = 0
        self.on_ticks = 0
        self.walk_session = sim.policy.ort_session
        self.log = log
        self.aligner = Aligner(foot)
        self.state = "NAVIGATE"
        self.t_state = 0.0
        self.pulses = 0
        self.squares = 0
        self.pulse_queue = []
        self.backoffs = 0
        self.pulse_left = 0
        self.settle_left = 0
        self.pulse_twist = (0.0, 0.0, 0.0)
        self.result = {"approach_mode": "precision" if precision_session is not None else "pulse"}
        b, _ = sim.ball()
        self.ball0 = b.copy()
        stance = self.stance_point(b)
        k = np.array([math.cos(self.kh), math.sin(self.kh)])
        line_up, pre = stance - LINE_UP_M * k, stance - PRE_STANCE_M * k
        (x, y, _), _, _ = sim.pose()
        route = [tuple(line_up), tuple(pre)]
        if self._seg_dist(np.array([x, y]), line_up, b) < DETOUR_CLEAR_M:
            side = np.array([-k[1], k[0]])
            side = side if (np.array([x, y]) - b) @ side > 0 else -side
            route.insert(0, tuple(b + 0.45 * side - 0.1 * k))
        self.nav = GotoSupervisor(route, ARENA, GotoConfig(arrival_radius=0.06), log=log.event)
        log.event({"t": 0.0, "event": "MISSION_START", "ball": [round(v, 3) for v in b],
                   "kick_heading_deg": round(math.degrees(kick_heading), 1),
                   "aim_heading_deg": round(math.degrees(self.kh), 1),
                   "foot": foot, "aim_offset_deg": round(math.degrees(KICK_AIM_OFFSET[foot]), 1),
                   "stance_shift_mm": round(PRECISE_STANCE_SHIFT_M[foot] * 1000, 1),
                   "stance": [round(v, 3) for v in stance], "route": [[round(v, 3) for v in p] for p in route]})

    def stance_point(self, ball_xy):
        """Duck position that puts the ball at the trained offset when facing the kick heading."""
        c, s = math.cos(self.kh), math.sin(self.kh)
        ox, oy = self.aligner.target
        return np.asarray(ball_xy) - np.array([c * ox - s * oy, s * ox + c * oy])

    @staticmethod
    def _seg_dist(a, b, p):
        ab = b - a
        t = np.clip((p - a) @ ab / max(ab @ ab, 1e-9), 0, 1)
        return float(np.linalg.norm(a + t * ab - p))

    def _goto(self, state):
        self.state, self.t_state = state, self.sim.t

    def _enter_fine(self, t):
        """Hand the last stretch to the precision policy if there is one (and entries left), else pulses."""
        if self.precision_session is not None and self.precise_entries < PRECISE_MAX_ENTRIES:
            self.settle_left = 20
            self._goto("PRECISE_PIVOT")
        else:
            self.log.event({"t": round(t, 3), "event": "ALIGN_START"})
            self.settle_left = ALIGN_SETTLE_TICKS
            self._goto("ALIGN")

    def _precise_fail(self, t, why):
        """Precision step gave up: VelStand back in, the pulse aligner (fallback) takes over."""
        self.sim.policy.ort_session = self.walk_session
        self.result["approach_mode"] = "precision->pulse fallback"
        self.result["precise_fail"] = why
        if why == "fall":
            self.result["falls"] = self.result.get("falls", 0) + 1
            self.down = True
        self.log.event({"t": round(t, 3), "event": "PRECISE_FALLBACK", "why": why})
        self.settle_left = ALIGN_SETTLE_TICKS
        self._goto("ALIGN")
        return (0.0, 0.0, 0.0)

    def precise_stance(self, ball_xy):
        """Stance target for the step policy: the trained stance, PRECISE_STANCE_SHIFT_M further along the kick line."""
        return self.stance_point(ball_xy) + PRECISE_STANCE_SHIFT_M[self.foot] * np.array([math.cos(self.kh), math.sin(self.kh)])

    def precise_offset(self, pose, ball_xy):
        """Option-A command: remaining offset to the stance pose in the trunk yaw frame, clipped to training range."""
        x, y, yaw = pose
        st = self.precise_stance(ball_xy)
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = c * (st[0] - x) + s * (st[1] - y), -s * (st[0] - x) + c * (st[1] - y)
        r = math.hypot(dx, dy)
        if r > PRECISE_MAX_R:
            dx, dy = dx * PRECISE_MAX_R / r, dy * PRECISE_MAX_R / r
        return (dx, dy, float(np.clip(wrap_angle(self.kh - yaw), -PRECISE_MAX_YAW, PRECISE_MAX_YAW)))

    def _swap_kick(self, t, pose, b, tilt, ex, ey, eh, aligned):
        sim = self.sim
        foot, lfoot = sim.site_local(sim.kick_site), sim.site_local(sim.support_site)
        vb = sim.data.qvel[sim.va:sim.va + 3]
        self.result.update(foot_vs_home_mm=[round(v, 1) for v in (foot - sim.home_foot) * 1000],
                           lfoot_vs_home_mm=[round(v, 1) for v in (lfoot - sim.home_lfoot) * 1000],
                           trunk_speed_mm_s=round(float(np.linalg.norm(vb[:2])) * 1000, 1),
                           tilt_deg=round(tilt, 1))
        q = np.array([sim.data.qpos[j] for j in sim.policy.joint_qpos_indices]) - sim.policy.default_pose
        # the kick only ever started from HOME +-0.05 rad; precision run 1 arrived crouched and never kicked
        self.result["leg_dev_deg"] = round(math.degrees(float(np.abs(q[[0, 1, 2, 3, 4, 9, 10, 11, 12, 13]]).max())), 1)
        self.result.update(align_ex_mm=round(ex * 1000, 1), align_ey_mm=round(ey * 1000, 1),
                           align_eh_deg=round(math.degrees(eh), 2), pulses=self.pulses,
                           aligned=aligned, t_swap=round(t, 2))
        self.log.event({"t": round(t, 3), "event": "KICK_SWAP_IN", "aligned": aligned, "pulses": self.pulses,
                        "mode": self.result["approach_mode"],
                        "err_mm": [round(ex * 1000, 1), round(ey * 1000, 1)], "err_heading_deg": round(math.degrees(eh), 2)})
        sim.policy.ort_session = self.kick_session
        self.kick_dir = np.array([math.cos(pose[2]), math.sin(pose[2])])  # what the kick policy aims at
        self.kick_ball0 = b.copy()
        self.peak = 0.0
        self._goto("KICK")
        return (0.0, 0.0, 0.0)

    def tick(self):
        """Returns the twist for this control tick (and swaps sessions as the state changes)."""
        sim, t = self.sim, self.sim.t
        pose, tilt, z = sim.pose()
        b, bv = sim.ball()
        if self.state in ("NAVIGATE", "ALIGN", "PRECISE_PIVOT", "PRECISE"):
            # A bumped ball: stand still until it stops rolling, then re-plan from where
            # it rests (ALIGN re-navigates if the new stance is out of reach). Chasing a
            # rolling ball with pulses never converges.
            if np.linalg.norm(bv) > 0.02 or np.linalg.norm(b - self.ball0) > 0.01:
                if not getattr(self, "ball_rolling", False):
                    self.ball_rolling, self.t_bump = True, t
                    self.pulse_left, self.approaching = 0, False
                    self.result["ball_disturbed"] = self.result.get("ball_disturbed", 0) + 1
                    sim.policy.ort_session = self.walk_session   # (a no-op unless the step policy had it)
                    self.log.event({"t": round(t, 3), "event": "BALL_DISTURBED", "phase": self.state})
                if np.linalg.norm(bv) > 0.02:
                    if t - self.t_bump < BALL_WAIT_S:
                        return (0.0, 0.0, 0.0)
                    # Still creeping after the wait (the sim ball has ~no rolling
                    # resistance): stop for good rather than chase it. Bounded failure.
                    self.result["aborted"] = "ball_lost"
                    self.log.event({"t": round(t, 3), "event": "BALL_LOST", "ball_speed": round(float(np.linalg.norm(bv)), 3)})
                    self._goto("DONE")
                    return (0.0, 0.0, 0.0)
                self.ball_rolling = False
                self.log.event({"t": round(t, 3), "event": "BALL_SETTLED",
                                "moved_m": round(float(np.linalg.norm(b - self.ball0)), 3)})
                self.ball0 = b.copy()
                self._enter_fine(t)
                return (0.0, 0.0, 0.0)

        if self.state == "NAVIGATE":
            self.nav.update(t, pose, tilt, z)
            twist = self.nav.command(t)
            if self.nav.state == "DONE":
                self._enter_fine(t)
                return (0.0, 0.0, 0.0)
            return twist

        if self.state == "PRECISE_PIVOT":   # VelStand turns onto the kick line first
            if tilt > 35 or z < 0.06:
                return self._precise_fail(t, "fall")
            eh = wrap_angle(pose[2] - self.kh)
            if abs(eh) < PRECISE_PIVOT_TOL or t - self.t_state > PRECISE_PIVOT_MAX_S:
                if self.settle_left > 0:
                    self.settle_left -= 1
                    return (0.0, 0.0, 0.0)
                self.precise_entries += 1
                self.on_ticks = 0
                sim.policy.ort_session = self.precision_session
                self.log.event({"t": round(t, 3), "event": "PRECISE_START", "n": self.precise_entries,
                                "offset": [round(v, 4) for v in self.precise_offset(pose, b)]})
                self._goto("PRECISE")
                return self.precise_offset(pose, b)
            self.settle_left = 20
            return (0.0, 0.0, -math.copysign(0.8, eh))

        if self.state == "PRECISE":
            if tilt > 35 or z < 0.06:
                return self._precise_fail(t, "fall")
            off = self.precise_offset(pose, b)
            vb = sim.data.qvel[sim.va:sim.va + 2]
            on = math.hypot(off[0], off[1]) < ALIGN_TOL and abs(off[2]) < ALIGN_HEADING_TOL
            self.on_ticks = self.on_ticks + 1 if on and np.linalg.norm(vb) < PRECISE_STILL_V else 0
            if self.on_ticks >= PRECISE_HOLD_TICKS:
                ex, ey, eh, _ = self.aligner.errors(pose, b, self.kh, sim.site_local(sim.kick_site), sim.home_foot)
                self.log.event({"t": round(t, 3), "event": "PRECISE_ARRIVED", "after_s": round(t - self.t_state, 2)})
                return self._swap_kick(t, pose, b, tilt, ex, ey, eh, aligned=True)
            if t - self.t_state > PRECISE_TIMEOUT_S:
                return self._precise_fail(t, "timeout")
            return off

        if self.state == "ALIGN":
            if tilt > 35 or z < 0.06:  # zero twist while down; VelStand gets up on its own
                if not getattr(self, "down", False):
                    self.down = True
                    self.pulse_left = 0
                    self.result["falls"] = self.result.get("falls", 0) + 1
                    self.log.event({"t": round(t, 3), "event": "FALL", "phase": "ALIGN"})
                self.settle_left = 50
                return (0.0, 0.0, 0.0)
            if getattr(self, "down", False):
                self.down = False
                self.log.event({"t": round(t, 3), "event": "RECOVERED", "phase": "ALIGN"})
            if getattr(self, "backing", False):
                ex, ey, eh, _ = self.aligner.errors(pose, b, self.kh)
                if ex > BACKOFF_TO or t - self.t_back > 3.0:
                    self.backing = False
                    self.settle_left = ALIGN_SETTLE_TICKS
                    self.log.event({"t": round(t, 3), "event": "ALIGN_BACKOFF_STOP", "err_mm": [round(ex * 1000, 1), round(ey * 1000, 1)]})
                    return (0.0, 0.0, 0.0)
                return clamp_align(-0.2, 0.0, 0.0)
            if getattr(self, "approaching", False):
                ex, ey, eh, _ = self.aligner.errors(pose, b, self.kh)  # trunk-relative while walking: the feet swing
                if ex < APPROACH_STOP or t - self.t_approach > 5.0:
                    self.approaching = False
                    self.settle_left = ALIGN_SETTLE_TICKS
                    self.log.event({"t": round(t, 3), "event": "ALIGN_APPROACH_STOP", "err_mm": [round(ex * 1000, 1), round(ey * 1000, 1)]})
                    return (0.0, 0.0, 0.0)
                # walk at cruise, steering at the stance spot (yaw-frame bearing to it)
                st = self.stance_point(b) - np.asarray(pose[:2])
                bearing = wrap_angle(math.atan2(st[1], st[0]) - pose[2])
                return clamp_align(0.2, 0.0, 2.0 * bearing)
            if self.pulse_left > 0:
                self.pulse_left -= 1
                if self.pulse_left == 0:
                    self.settle_left = ALIGN_SETTLE_TICKS
                return clamp_align(*self.pulse_twist)
            if self.settle_left > 0:
                self.settle_left -= 1
                return (0.0, 0.0, 0.0)
            if self.pulse_queue:  # rest of a multi-pulse squaring move
                self.pulse_twist, self.pulse_left = self.pulse_queue.pop(0)
                self.pulse_left -= 1
                return clamp_align(*self.pulse_twist)
            ex, ey, eh, local = self.aligner.errors(pose, b, self.kh, sim.site_local(sim.kick_site), sim.home_foot)
            if ex > APPROACH_FROM and abs(eh) < math.radians(10) and abs(ey) < 0.03:
                self.approaching, self.t_approach = True, t
                self.log.event({"t": round(t, 3), "event": "ALIGN_APPROACH", "err_mm": [round(ex * 1000, 1), round(ey * 1000, 1)]})
                return (0.0, 0.0, 0.0)
            stance = self.stance_point(b)
            if np.linalg.norm(stance - pose[:2]) > ALIGN_ZONE_M:   # drifted out: navigate back in
                self.log.event({"t": round(t, 3), "event": "ALIGN_LOST", "dist": round(float(np.linalg.norm(stance - pose[:2])), 3)})
                k = np.array([math.cos(self.kh), math.sin(self.kh)])
                self.nav = GotoSupervisor([tuple(stance - PRE_STANCE_M * k)], ARENA, GotoConfig(arrival_radius=0.06),
                                          log=self.log.event)
                self._goto("NAVIGATE")
                return (0.0, 0.0, 0.0)
            p = self.aligner.pulse(ex, ey, eh, allow_backoff=self.backoffs < MAX_BACKOFFS)
            if p is not None and p[2] == "backoff" and self.pulses < ALIGN_MAX_PULSES:
                self.pulses += 1
                self.backoffs += 1
                self.backing, self.t_back = True, t
                self.log.event({"t": round(t, 3), "event": "ALIGN_BACKOFF", "err_mm": [round(ex * 1000, 1), round(ey * 1000, 1)]})
                return (0.0, 0.0, 0.0)
            if p is not None and p[2] == "backoff":
                p = None  # out of pulses: kick from here
            stagger = (sim.site_local(sim.kick_site) - sim.home_foot)[0]
            support = (sim.site_local(sim.support_site) - sim.home_lfoot)[0]
            if (p is None and (abs(stagger) > SQUARE_TOL or abs(support) > SUPPORT_TOL)
                    and self.squares < SQUARE_MAX and self.pulses < ALIGN_MAX_PULSES):
                # Ball placement is fine but a foot sits too far from its HOME spot: the
                # kick only ever started from a square stance. Measured over 133 mission
                # kicks: all 7 whiffs had the feet split front-to-back (kicking foot 22-33 mm
                # behind HOME, or the support foot 13-16 mm behind, where the policy doesn't
                # swing at all). Re-square, then re-check the fit.
                self.squares += 1
                self.pulses += 1
                (self.pulse_twist, self.pulse_left), *self.pulse_queue = SQUARE_MOVE
                self.log.event({"t": round(t, 3), "event": "ALIGN_SQUARE", "n": self.squares,
                                "kick_foot_vs_home_mm": round(stagger * 1000, 1),
                                "support_foot_vs_home_mm": round(support * 1000, 1)})
                self.pulse_left -= 1
                return clamp_align(*self.pulse_twist)
            if p is None or self.pulses >= ALIGN_MAX_PULSES:
                return self._swap_kick(t, pose, b, tilt, ex, ey, eh, aligned=p is None)
            self.pulse_twist, self.pulse_left, axis = p
            self.pulses += 1
            self.log.event({"t": round(t, 3), "event": "ALIGN_PULSE", "n": self.pulses, "axis": axis, "ticks": self.pulse_left,
                            "err_mm": [round(ex * 1000, 1), round(ey * 1000, 1)], "err_heading_deg": round(math.degrees(eh), 2)})
            self.pulse_left -= 1
            return clamp_align(*self.pulse_twist)

        _, bv = sim.ball()
        if self.state == "KICK":
            fwd = float(bv @ np.array([math.cos(self.kh_goal), math.sin(self.kh_goal)]))
            if "kick_contact_t" not in self.result and np.linalg.norm(bv) > 0.05:
                self.result["kick_contact_t"] = round(t - self.t_state, 2)
                self.log.event({"t": round(t, 3), "event": "KICK_FIRE", "after_swap_s": self.result["kick_contact_t"]})
            self.peak = max(self.peak, fwd)
            if t - self.t_state >= KICK_HOLD_S:
                sim.policy.ort_session = self.walk_session
                self.log.event({"t": round(t, 3), "event": "KICK_SWAP_OUT", "peak_ball_fwd": round(self.peak, 3)})
                self.fell = False
                self._goto("SETTLE")
            return (0.0, 0.0, 0.0)

        if self.state == "SETTLE":
            self.peak = max(self.peak, float(bv @ np.array([math.cos(self.kh_goal), math.sin(self.kh_goal)])))
            if (tilt > 35 or z < 0.06) and not self.fell:
                self.fell = True
                self.result["falls"] = self.result.get("falls", 0) + 1
                self.log.event({"t": round(t, 3), "event": "FALL", "phase": "SETTLE"})
            up = tilt < 25 and z > 0.09
            if self.fell and up and not self.result.get("recovered"):
                self.result["recovered"] = True
                self.log.event({"t": round(t, 3), "event": "RECOVERED"})
            if t - self.t_state >= SETTLE_S and up:
                d = b - self.kick_ball0
                self.result.update(peak_ball_fwd=round(self.peak, 3), ball_travel=round(float(np.linalg.norm(d)), 3),
                                   kick_dir_err_deg=round(math.degrees(wrap_angle(math.atan2(d[1], d[0]) - self.kh_goal)), 1)
                                   if np.linalg.norm(d) > 0.02 else None, t_done=round(t, 2))
                self.log.event({"t": round(t, 3), "event": "MISSION_DONE", **self.result})
                self._goto("DONE")
        return (0.0, 0.0, 0.0)


def draw_mission(scn, m, sim):
    """Stance target (disk = +-15 mm), kick-direction arrow from the ball, pending route."""
    scn.ngeom = 0

    def add(gtype, size, pos, rgba):
        if scn.ngeom >= scn.maxgeom:
            return None
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float), np.eye(3).flatten(),
                            np.asarray(rgba, np.float32))
        scn.ngeom += 1
        return g

    def line(a, b, rgba, width=0.004):
        g = add(mujoco.mjtGeom.mjGEOM_CAPSULE, [1, 1, 1], [0, 0, 0], rgba)
        if g is not None:
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a, float), np.asarray(b, float))

    if m.state == "DONE":
        return
    b, _ = sim.ball()
    k = np.array([math.cos(m.kh_goal), math.sin(m.kh_goal)])   # the intended kick line (the duck aims off it)
    line((*b, 0.004), (*(b + 0.35 * k), 0.004), [1.0, 0.85, 0.1, 0.9], 0.005)
    tip = b + 0.35 * k
    side = np.array([-k[1], k[0]])
    for s in (1, -1):
        line((*tip, 0.004), (*(tip - 0.06 * k + s * 0.035 * side), 0.004), [1.0, 0.85, 0.1, 0.9], 0.005)
    st = m.precise_stance(b) if m.precision_session is not None else m.stance_point(b)
    add(mujoco.mjtGeom.mjGEOM_CYLINDER, [0.03, 0.002, 0], [*st, 0.002], [0.1, 0.9, 0.2, 0.6])
    if m.state == "NAVIGATE":
        (px, py, _), _, _ = sim.pose()
        prev = (px, py)
        for _, (x, y) in m.nav.queue:
            add(mujoco.mjtGeom.mjGEOM_CYLINDER, [0.03, 0.002, 0], [x, y, 0.002], [1.0, 0.6, 0.1, 0.5])
            line((*prev, 0.006), (x, y, 0.006), [1.0, 0.6, 0.1, 0.5], 0.002)
            prev = (x, y)


def new_mission(sim, kick, rng, log, ball=None, kick_heading=None, precision=None, foot="right"):
    """Put the ball 1.0-1.8 m from the duck (or at `ball`) and start a mission from where the duck stands."""
    (x, y, _), _, _ = sim.pose()
    if ball is None:
        for _ in range(100):
            d, a = rng.uniform(1.0, 1.8), rng.uniform(-math.pi, math.pi)
            ball = (x + d * math.cos(a), y + d * math.sin(a))
            if ARENA[0] + 0.6 < ball[0] < ARENA[1] - 0.6 and ARENA[2] + 0.6 < ball[1] < ARENA[3] - 0.6:
                break
    kh = rng.uniform(-math.pi, math.pi) if kick_heading is None else kick_heading
    sim.data.qpos[sim.bqa:sim.bqa + 7] = [ball[0], ball[1], 0.035, 1, 0, 0, 0]
    sim.data.qvel[sim.bva:sim.bva + 6] = 0
    mujoco.mj_forward(sim.model, sim.data)
    return Mission(sim, kick, kh, log, foot=foot, precision_session=precision)


def set_feet(sim, foot):
    """Kicking/support foot sites and their HOME positions (trunk yaw frame), read at the reset stand."""
    sim.kick_site, sim.support_site = f"{foot}_foot", ("left_foot" if foot == "right" else "right_foot")
    sim.home_foot, sim.home_lfoot = sim.site_local(sim.kick_site), sim.site_local(sim.support_site)


def run_viewer(args):
    import mujoco.viewer
    sim = KickSim(args.velstand, SCENES[args.scene])
    kick = ort.InferenceSession(args.kick)
    precision = ort.InferenceSession(args.precision) if args.precision else None
    sim.reset(0, 0, 0)
    set_feet(sim, args.foot)
    rng = np.random.default_rng(args.seed)
    out = f"logs/kick_mission/{time.strftime('%Y-%m-%d_%H-%M-%S')}_viewer"
    os.makedirs(out, exist_ok=True)
    ball = tuple(float(v) for v in args.ball.split(",")) if args.ball else None
    kh = math.radians(args.kick_heading) if args.kick_heading is not None else None
    log = RunLog(echo=not args.auto)
    m = new_mission(sim, kick, rng, log, ball, kh, precision, args.foot)
    n, t_mission, t_done, tally = 0, sim.t, None, {"kicked": 0, "missed": 0, "aborted": 0, "falls": 0}

    def report(m):
        r = m.result
        if r.get("aborted") or m.state != "DONE":
            tally["aborted"] += 1
            outcome = f"ABORTED ({r.get('aborted', 'timeout in ' + m.state)})"
        elif (r.get("peak_ball_fwd") or 0) > 0.05:
            tally["kicked"] += 1
            de = r.get("kick_dir_err_deg")
            outcome = (f"KICKED  ball {r['peak_ball_fwd']:.2f} m/s, rolled {r.get('ball_travel', 0):.2f} m"
                       + (f", {de:+.0f} deg off the arrow" if de is not None else ""))
        else:
            tally["missed"] += 1
            outcome = "MISSED"
        tally["falls"] += r.get("falls", 0)
        mode = {"precision": "step", "precision->pulse fallback": "step->pulse", "pulse": "pulse"}[r["approach_mode"]]
        print(f"  mission {n + 1:2d}: {outcome:58s} [{mode}] aligned to ({r.get('align_ex_mm', float('nan')):+.0f}, "
              f"{r.get('align_ey_mm', float('nan')):+.0f}) mm in {r.get('pulses', '-')} pulses   "
              f"[tally: {tally['kicked']} kicked, {tally['missed']} missed, {tally['aborted']} aborted, "
              f"{tally['falls']} falls]", flush=True)

    print("\nKeys (type in THIS terminal): N = new mission (ball re-placed), P = push, Q = quit"
          + (f"\nAuto mode: {args.auto} missions back to back\n" if args.auto else "\n"))
    with ip.TerminalInput() as term, \
            mujoco.viewer.launch_passive(sim.model, sim.data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = sim.trunk_body
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 1.2, -30, 135
        quit_req = False
        while viewer.is_running() and not quit_req:
            t0 = time.time()
            for key in term.get_keys():
                if key == "q":
                    quit_req = True
                elif key == "p":
                    a = rng.uniform(0, 2 * math.pi)
                    sim.push(1.0 * math.cos(a), 1.0 * math.sin(a))
                    log.event({"t": round(sim.t, 3), "event": "PUSH"})
                elif key == "n":
                    sim.policy.ort_session = m.walk_session
                    log.save(f"{out}/mission{n}")
                    n += 1
                    log = RunLog()
                    m = new_mission(sim, kick, rng, log, precision=precision, foot=args.foot)
            if args.auto:
                over = m.state == "DONE" or sim.t - t_mission > 60.0
                if over and t_done is None:
                    t_done = sim.t
                    sim.policy.ort_session = m.walk_session
                    report(m)
                if t_done is not None and sim.t - t_done > 1.5:  # a beat of standing between missions
                    log.save(f"{out}/mission{n}")
                    n += 1
                    if n >= args.auto:
                        print(f"\n  {args.auto} missions: {tally['kicked']} kicked, {tally['missed']} missed, "
                              f"{tally['aborted']} aborted, {tally['falls']} falls")
                        break
                    log = RunLog(echo=False)
                    m = new_mission(sim, kick, rng, log, precision=precision, foot=args.foot)
                    t_mission, t_done = sim.t, None
            twist = m.tick() if m.state not in ("DONE",) and t_done is None else (0.0, 0.0, 0.0)
            sim.set_twist(twist)
            sim.step()
            draw_mission(viewer.user_scn, m, sim)
            viewer.sync()
            time.sleep(max(0.0, CONTROL_DT - (time.time() - t0)))
    log.save(f"{out}/mission{n}")
    print(f"logs: {out}/")


def run_headless(args):
    sim = KickSim(args.velstand, SCENES[args.scene])
    kick = ort.InferenceSession(args.kick)
    precision = ort.InferenceSession(args.precision) if args.precision else None
    sim.reset(0, 0, 0)
    set_feet(sim, args.foot)
    print(f"{args.foot} (kicking) foot at HOME, trunk yaw frame: ({sim.home_foot[0]*1000:.1f}, {sim.home_foot[1]*1000:.1f}) mm")
    rng = np.random.default_rng(args.seed)
    out = f"logs/kick_mission/{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(out, exist_ok=True)
    print(f"velstand={args.velstand}\nkick={args.kick}\nprecision={args.precision} (pulse aligner = fallback)\nscene={args.scene}  logs -> {out}/")
    print(f"kicking foot {args.foot}: precision stance shift {PRECISE_STANCE_SHIFT_M[args.foot]*1000:+.1f} mm along the kick line; "
          f"aim offset {math.degrees(KICK_AIM_OFFSET[args.foot]):+.1f} deg (+ = left)")
    rows = []
    for k in range(args.runs):
        sim.reset(0.0, 0.0, rng.uniform(-math.pi, math.pi))
        bd, bb = rng.uniform(1.0, 1.8), rng.uniform(-math.pi, math.pi)
        kh = rng.uniform(-math.pi, math.pi)
        ball = (bd * math.cos(bb), bd * math.sin(bb))
        sim.data.qpos[sim.bqa:sim.bqa + 7] = [*ball, 0.035, 1, 0, 0, 0]
        sim.data.qvel[sim.bva:sim.bva + 6] = 0
        mujoco.mj_forward(sim.model, sim.data)
        log = RunLog(echo=args.verbose)
        m = Mission(sim, kick, kh, log, foot=args.foot, precision_session=precision)
        while m.state != "DONE" and sim.t < args.timeout:
            twist = m.tick()
            sim.set_twist(twist)
            (x, y, yaw), tilt, z = sim.pose()
            bxy, _ = sim.ball()
            rf, lf = sim.site_local("right_foot"), sim.site_local("left_foot")
            log.trace.append({"t": round(sim.t, 3), "state": m.state, "x": round(x, 4), "y": round(y, 4),
                              "yaw": round(yaw, 4), "tilt_deg": round(tilt, 1), "trunk_z": round(z, 4),
                              "cmd_vx": round(twist[0], 3), "cmd_vy": round(twist[1], 3), "cmd_wz": round(twist[2], 3),
                              "ball_x": round(float(bxy[0]), 4), "ball_y": round(float(bxy[1]), 4),
                              "rfoot_x": round(float(rf[0]), 4), "rfoot_y": round(float(rf[1]), 4),
                              "lfoot_x": round(float(lf[0]), 4), "lfoot_y": round(float(lf[1]), 4)})
            sim.step()
        if m.state != "DONE":
            sim.policy.ort_session = m.walk_session
            m.result["timeout_in"] = m.state
        r = {"run": k, "ball_dist_m": round(bd, 2), **m.result}
        rows.append(r)
        log.save(f"{out}/run{k}")
        print(f"  run {k}: ball {bd:.2f} m away, kick heading {math.degrees(kh):+4.0f}deg -> " +
              ", ".join(f"{a}={b}" for a, b in r.items() if a not in ("run", "ball_dist_m")), flush=True)
    al = [r for r in rows if r.get("aligned")]
    print(f"\n  aligned within +-{ALIGN_TOL*1000:.0f} mm / {math.degrees(ALIGN_HEADING_TOL):.0f} deg: {len(al)}/{len(rows)}; "
          f"kicked (peak > 0.05 m/s): {sum((r.get('peak_ball_fwd') or 0) > 0.05 for r in rows)}/{len(rows)}; "
          f"falls: {sum(r.get('falls', 0) for r in rows)}; ball disturbed before kick: "
          f"{sum(r.get('ball_disturbed', 0) > 0 for r in rows)} runs; median mission time "
          f"{np.median([r['t_done'] for r in rows if 't_done' in r]) if any('t_done' in r for r in rows) else float('nan'):.1f} s")
    modes = {}
    for r in rows:
        modes[r["approach_mode"]] = modes.get(r["approach_mode"], 0) + 1
    print(f"  approach mode: {modes}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("velstand")
    ap.add_argument("kick")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scene", choices=list(SCENES), default="allcollisions_ball")
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--ball", default=None, help='viewer: "x,y" ball position (default: random 1.0-1.8 m away)')
    ap.add_argument("--kick-heading", type=float, default=None, help="viewer: kick direction, deg (default: random)")
    ap.add_argument("--auto", type=int, default=0, help="viewer: run this many missions back to back, with a tally")
    ap.add_argument("--foot", choices=("right", "left"), default="right",
                    help="kicking foot of the kick ONNX (sets the stance side and that foot's calibrations)")
    ap.add_argument("--precision", default=None,
                    help="PrecisionStep ONNX for the final approach (pulse aligner stays as the fallback)")
    args = ap.parse_args()
    run_viewer(args) if args.viewer else run_headless(args)


if __name__ == "__main__":
    main()

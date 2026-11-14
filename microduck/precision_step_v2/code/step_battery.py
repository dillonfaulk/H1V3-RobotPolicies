"""Headless placement battery for a PrecisionStep ONNX (CPU MuJoCo, eval-harness settings).

Run from the microduck_rl repo root:
  uv run python scripts/step_battery.py <precision_step.onnx> [--episodes 50]
      [--prestand <velstand.onnx>] [--push-at 1.5 --push-v 0.3] [--ball-prob 1.0]
      [--ball-side right|left|both]

--ball-side (PrecisionStep v2): which foot's kick spot the ball sits at. "right"
(default) is v1's battery, rng stream unchanged; "both" draws the side 50/50 per
episode and the summary splits every metric by side.

Each episode mirrors the training reset: standing start (HOME joints +-0.08 rad,
+-5 deg roll/pitch, trunk z 0.11-0.12, random yaw), a target pose drawn from the
training distribution (mostly < 80 mm, tail to 250 mm, +-15 deg heading, 10%
"hold still"), and the ball at the right-foot kick spot relative to the TARGET
stance whenever it clears the duck's start footprint. Every tick the 3 twist
slots get the remaining offset [dx, dy, dyaw] in the trunk's yaw frame (the
option-A contract). 6 s episodes.

Per episode: final position / heading error, front-back foot stagger, whether
the duck is still at the end, falls, ball strikes (> 1 cm displacement, the
training failure rule), and KICK-READY = within 12 mm / 3 deg, stagger < 10 mm,
still, every leg joint within 0.10 rad of HOME (the kick's start stance), never
struck, never fell.
"""
import argparse
import csv
import math
import os
import sys
import time

import mujoco
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from goto_point import CONTROL_DT  # noqa: E402
from kick_battery import BALL_RADIUS, FALL_TILT_DEG, FALL_Z, KickSim  # noqa: E402

SCENE = "src/mjlab_microduck/robot/microduck/scene_allcollisions_ball.xml"
EPISODE_S = 6.0
# microduck_precision_step_env_cfg
NEAR_R, FAR_R, FAR_PROB, CONE_PROB, CONE_HALF = (0.0, 0.08), (0.08, 0.25), 0.35, 0.5, math.pi / 4
YAW_RANGE, ZERO_PROB = math.radians(15), 0.1
BALL_OFFSET, BALL_NOISE = (0.09, -0.042), 0.015
FOOT_X, FOOT_Y, KEEPOUT_PAD = (-0.033, 0.034), (-0.079, 0.079), BALL_RADIUS + 0.006
STRUCK = 0.01
POS_TOL, YAW_TOL, STAGGER_TOL = 0.012, math.radians(3), 0.010
STILL_V, STILL_W, STILL_WINDOW = 0.02, 0.2, 0.5   # m/s, rad/s, over the last 0.5 s
# The kick only ever started from HOME +-0.05 rad (+ encoder bias). Run-1 model_1000
# reached every target in a ~30-36 deg hip/knee crouch and the kick never swung
# (0/60 missions), so posture is part of KICK-READY.
LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
POSE_TOL = 0.10   # rad, every leg joint vs HOME


def sample_target(rng):
    r = rng.uniform(*FAR_R) if rng.random() < FAR_PROB else rng.uniform(*NEAR_R)
    b = rng.uniform(-CONE_HALF, CONE_HALF) if rng.random() < CONE_PROB else rng.uniform(-math.pi, math.pi)
    dyaw = rng.uniform(-YAW_RANGE, YAW_RANGE)
    if rng.random() < ZERO_PROB:
        r, dyaw = 0.0, 0.0
    return r * math.cos(b), r * math.sin(b), dyaw


def to_frame(dx, dy, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class StepSim(KickSim):
    def reset_step_episode(self, rng, ball_prob=1.0, target=None):
        yaw = rng.uniform(-math.pi, math.pi)
        self.reset(0.0, 0.0, yaw)
        d, qa, p = self.data, self.qa, self.policy
        roll, pitch = rng.uniform(-math.radians(5), math.radians(5), 2)
        cr, sr, cp, sp = math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2), math.sin(pitch / 2)
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        d.qpos[qa + 2] = rng.uniform(0.11, 0.12)
        d.qpos[qa + 3:qa + 7] = [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                                 cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]
        for i, q in enumerate(p.joint_qpos_indices):
            d.qpos[q] = p.default_pose[i] + rng.uniform(-0.08, 0.08)
        self.bam.reset(d.qpos)
        p.ort_session = self.kick_session          # the precision policy (KickSim's "kick" slot)
        self.set_goal(rng, ball_prob, target)

    ball_side = "right"   # "right" | "left" | "both" (set by main)

    def set_goal(self, rng, ball_prob=1.0, target=None):
        """Target pose relative to the CURRENT pose; ball at the kick spot from it if clear."""
        (x0, y0, yaw0), _, _ = self.pose()
        dx, dy, dyaw = target if target is not None else sample_target(rng)
        side = self.ball_side
        if side == "both":
            side = "left" if rng.random() < 0.5 else "right"
        self.side = side
        c, s = math.cos(yaw0), math.sin(yaw0)
        self.goal = (x0 + c * dx - s * dy, y0 + s * dx + c * dy, wrap(yaw0 + dyaw))
        self.goal_r = math.hypot(dx, dy)
        ox = BALL_OFFSET[0] + rng.uniform(-BALL_NOISE, BALL_NOISE)
        oy = (abs(BALL_OFFSET[1]) if side == "left" else BALL_OFFSET[1]) + rng.uniform(-BALL_NOISE, BALL_NOISE)
        gx, gy, gyaw = self.goal
        bx = gx + math.cos(gyaw) * ox - math.sin(gyaw) * oy
        by = gy + math.sin(gyaw) * ox + math.cos(gyaw) * oy
        lx, ly = to_frame(bx - x0, by - y0, yaw0)
        inside = (FOOT_X[0] - KEEPOUT_PAD < lx < FOOT_X[1] + KEEPOUT_PAD) and (FOOT_Y[0] - KEEPOUT_PAD < ly < FOOT_Y[1] + KEEPOUT_PAD)
        self.has_ball = (rng.random() < ball_prob) and not inside
        if not self.has_ball:
            bx, by = x0 + 3.0, y0
        self.data.qpos[self.bqa:self.bqa + 7] = [bx, by, BALL_RADIUS, 1, 0, 0, 0]
        self.data.qvel[self.bva:self.bva + 6] = 0
        mujoco.mj_forward(self.model, self.data)
        self.ball_home = np.array([bx, by])

    def offset(self):
        (x, y, yaw), _, _ = self.pose()
        bx, by = to_frame(self.goal[0] - x, self.goal[1] - y, yaw)
        return bx, by, wrap(self.goal[2] - yaw)

    def stagger(self):
        lf, rf = self.site_local("left_foot"), self.site_local("right_foot")
        return float(lf[0] - rf[0]), float(lf[1] - rf[1])


def run_episode(sim, rng, ball_prob=1.0, prestand=None, prestand_s=1.5, push=None):
    if prestand is not None:
        sim.reset_step_episode(rng, ball_prob=0.0)
        sim.policy.ort_session = prestand
        sim.set_twist((0.0, 0.0, 0.0))
        while sim.t < prestand_s - 1e-9:
            sim.step()
        sim.set_goal(rng, ball_prob)          # target + ball from the settled VelStand stance
        sim.policy.ort_session = sim.kick_session
        sim.t = 0.0
    else:
        sim.reset_step_episode(rng, ball_prob)
    struck_t, fell_t, arrive_t, trail = None, None, None, []
    while sim.t < EPISODE_S - 1e-9:
        sim.set_twist(sim.offset())
        if push is not None and abs(sim.t - push[0]) < CONTROL_DT / 2:
            a = rng.uniform(0, 2 * math.pi)
            sim.push(push[1] * math.cos(a), push[1] * math.sin(a))
        sim.step()
        dx, dy, dyaw = sim.offset()
        e = math.hypot(dx, dy)
        on = e < POS_TOL and abs(dyaw) < YAW_TOL
        arrive_t = (arrive_t if arrive_t is not None else sim.t) if on else None
        if sim.has_ball and struck_t is None:
            bxy, _ = sim.ball()
            if np.linalg.norm(bxy - sim.ball_home) > STRUCK:
                struck_t = sim.t
        _, tilt, z = sim.pose()
        if fell_t is None and (tilt > FALL_TILT_DEG or z < FALL_Z):
            fell_t = sim.t
        v = sim.data.qvel[sim.va:sim.va + 2]
        trail.append((float(np.hypot(*v)), abs(float(sim.data.qvel[sim.va + 5]))))
    dx, dy, dyaw = sim.offset()
    sx, sy = sim.stagger()
    k = int(round(STILL_WINDOW / CONTROL_DT))
    still = max(t[0] for t in trail[-k:]) < STILL_V and max(t[1] for t in trail[-k:]) < STILL_W
    pos_err, yaw_err = math.hypot(dx, dy), abs(dyaw)
    q = np.array([sim.data.qpos[j] for j in sim.policy.joint_qpos_indices]) - sim.policy.default_pose
    leg_dev = float(np.abs(q[LEG_JOINTS]).max())
    ready = (pos_err < POS_TOL and yaw_err < YAW_TOL and abs(sx) < STAGGER_TOL and still
             and leg_dev < POSE_TOL and struck_t is None and fell_t is None)
    return {"target_r_mm": round(sim.goal_r * 1000, 1), "has_ball": sim.has_ball,
            "pos_err_mm": round(pos_err * 1000, 1), "yaw_err_deg": round(math.degrees(yaw_err), 2),
            "stagger_mm": round(sx * 1000, 1), "width_mm": round(sy * 1000, 1), "still": still, "leg_dev_deg": round(math.degrees(leg_dev), 1),
            "arrive_t": arrive_t, "struck_t": struck_t, "fell_t": fell_t, "kick_ready": ready,
            "ball_side": sim.side}


def pct(xs, q):
    return float(np.percentile(xs, q)) if xs else float("nan")


def bucket(r):
    return "hold (0)" if r == 0 else ("near (<80 mm)" if r < 80 else "far (80-250 mm)")


def summarize(rows, label):
    n = len(rows)
    up = [r for r in rows if r["fell_t"] is None]
    pe = [r["pos_err_mm"] for r in up]
    ye = [r["yaw_err_deg"] for r in up]
    sg = [abs(r["stagger_mm"]) for r in up]
    balls = [r for r in rows if r["has_ball"]]
    ld = [r["leg_dev_deg"] for r in up]
    lines = [f"== {label}: {n} episodes ({len(balls)} with a ball)",
             f"  KICK-READY (<= 12 mm, 3 deg, stagger < 10 mm, still, legs within 0.10 rad of HOME, no strike, no fall): "
             f"{sum(r['kick_ready'] for r in rows)}/{n}",
             f"  final position error: median {pct(pe, 50):.1f} mm, p90 {pct(pe, 90):.1f}, worst {max(pe) if pe else float('nan'):.1f}",
             f"  final heading error: median {pct(ye, 50):.2f} deg, p90 {pct(ye, 90):.2f}, worst {max(ye) if ye else float('nan'):.2f}",
             f"  front-back foot stagger: median {pct(sg, 50):.1f} mm, p90 {pct(sg, 90):.1f}, worst {max(sg) if sg else float('nan'):.1f}; "
             f"< 10 mm in {sum(s < 10 for s in sg)}/{len(sg)}",
             f"  still at the end: {sum(r['still'] for r in up)}/{len(up)}",
             f"  end posture, worst leg joint vs HOME: median {pct(ld, 50):.1f} deg, worst {max(ld) if ld else float('nan'):.1f}; "
             f"within {math.degrees(POSE_TOL):.1f} deg in {sum(x < math.degrees(POSE_TOL) for x in ld)}/{len(ld)}",
             f"  ball struck: {sum(r['struck_t'] is not None for r in balls)}/{len(balls)} ball episodes",
             f"  falls: {n - len(up)}/{n}"]
    for b in ("hold (0)", "near (<80 mm)", "far (80-250 mm)"):
        rs = [r for r in rows if bucket(r["target_r_mm"]) == b]
        if rs:
            ups = [r["pos_err_mm"] for r in rs if r["fell_t"] is None]
            lines.append(f"    {b:16s} {len(rs):3d} eps: kick-ready {sum(r['kick_ready'] for r in rs)}, "
                         f"median err {pct(ups, 50):.1f} mm, struck {sum(r['struck_t'] is not None for r in rs)}, "
                         f"falls {sum(r['fell_t'] is not None for r in rs)}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--ball-prob", type=float, default=1.0, help="ball whenever it clears the start footprint")
    ap.add_argument("--prestand", default=None, help="VelStand ONNX that holds the stand before the swap-in")
    ap.add_argument("--prestand-s", type=float, default=1.5)
    ap.add_argument("--push-at", type=float, default=None)
    ap.add_argument("--push-v", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ball-side", choices=("right", "left", "both"), default="right")
    ap.add_argument("--out", default=None, help="csv path for per-episode rows")
    args = ap.parse_args()

    sim = StepSim(args.onnx, SCENE)
    sim.ball_side = args.ball_side
    prestand = ort.InferenceSession(args.prestand) if args.prestand else None
    push = (args.push_at, args.push_v) if args.push_at is not None else None
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    rows = [run_episode(sim, rng, args.ball_prob, prestand, args.prestand_s, push) for _ in range(args.episodes)]
    label = (f"{os.path.basename(args.onnx)}"
             + (f", swapped in after {args.prestand_s}s of VelStand standing" if prestand else ", training-like start")
             + (f", push {args.push_v} m/s at {args.push_at}s" if push else ""))
    label += f", ball side {args.ball_side}"
    print(summarize(rows, label))
    if args.ball_side == "both":
        for side in ("right", "left"):
            print(summarize([r for r in rows if r["ball_side"] == side], f"ball on the {side.upper()}"))
    print(f"  ({time.time() - t0:.0f} s wall)")
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()

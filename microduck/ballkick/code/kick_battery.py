"""Headless kick battery for a BallKick ONNX (CPU MuJoCo, eval-harness settings).

Run from the microduck_rl repo root:
  uv run python scripts/kick_battery.py <kick.onnx> [--episodes 50] [--target 1.0]
      [--scene ball|allcollisions_ball] [--handoff <velstand.onnx> --handoff-at 2.0]
      [--push-at 3.0 --push-v 0.3]

Each episode mirrors the training reset: standing start (HOME joints +-0.05 rad,
+-5 deg roll/pitch, trunk z 0.11-0.12, random yaw), ball at the kicking foot's
offset + uniform +-15 mm noise, all-zero command, 5 s episode. Per episode it
reports ball speed vs the target, travel, direction error vs the reset heading,
and how the duck ended up. With --handoff the kick policy runs for --handoff-at
seconds and VelStand takes over (the mission's swap-back), which is what the
stay-up rate should be judged on for deployment.
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
from goto_point import CONTROL_DT, DuckSim  # noqa: E402

SCENES = {"ball": "src/mjlab_microduck/robot/microduck/scene_ball.xml",
          "allcollisions_ball": "src/mjlab_microduck/robot/microduck/scene_allcollisions_ball.xml"}
# microduck_ball_kick_env_cfg
BALL_OFFSET_X, BALL_OFFSET_ABS_Y, BALL_RADIUS, BALL_NOISE = 0.09, 0.042, 0.035, 0.015
EPISODE_S = 5.0
FALL_TILT_DEG, FALL_Z = 70.0, 0.06      # training fell_over
UP_TILT_DEG, UP_Z = 25.0, 0.09          # VelStand "recovered"
MOVED = 0.05                            # m/s: the ball has been touched


class KickSim(DuckSim):
    def __init__(self, onnx, scene, delay=1):
        super().__init__(onnx, scene=scene, delay=delay)
        j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        self.bqa, self.bva = self.model.jnt_qposadr[j], self.model.jnt_dofadr[j]
        self.kick_session = self.policy.ort_session

    def reset_episode(self, rng, foot="right", ball_offset=None):
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
            d.qpos[q] = p.default_pose[i] + rng.uniform(-0.05, 0.05)
        self.bam.reset(d.qpos)
        p.ort_session = self.kick_session
        if ball_offset is None:
            oy = -BALL_OFFSET_ABS_Y if foot == "right" else BALL_OFFSET_ABS_Y
            ball_offset = (BALL_OFFSET_X + rng.uniform(-BALL_NOISE, BALL_NOISE), oy + rng.uniform(-BALL_NOISE, BALL_NOISE))
        self.place_ball(ball_offset, yaw)
        self.heading = yaw
        return ball_offset

    def place_ball(self, offset_local, yaw):
        """Ball center at offset_local in the robot's yaw frame (training's reset_ball_in_front_of_foot)."""
        x, y = self.data.qpos[self.qa], self.data.qpos[self.qa + 1]
        bx = x + math.cos(yaw) * offset_local[0] - math.sin(yaw) * offset_local[1]
        by = y + math.sin(yaw) * offset_local[0] + math.cos(yaw) * offset_local[1]
        self.data.qpos[self.bqa:self.bqa + 7] = [bx, by, BALL_RADIUS, 1, 0, 0, 0]
        self.data.qvel[self.bva:self.bva + 6] = 0
        mujoco.mj_forward(self.model, self.data)

    def site_local(self, name):
        """A site's xy in the trunk's yaw frame (e.g. 'right_foot')."""
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        (x, y, yaw), _, _ = self.pose()
        d = self.data.site_xpos[sid][:2] - (x, y)
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])

    def ball(self):
        return self.data.qpos[self.bqa:self.bqa + 2].copy(), self.data.qvel[self.bva:self.bva + 2].copy()


def run_episode(sim, rng, handoff=None, handoff_at=2.0, push=None, prestand=None, prestand_s=1.5,
                reset_last_action=False):
    sim.reset_episode(rng)
    if prestand is not None:
        # Mission-style start: VelStand holds the stand first, then the ball goes
        # down at the trained offset (+ noise) in the settled yaw frame and the
        # kick policy is hard-swapped in.
        sim.policy.ort_session = prestand
        while sim.t < prestand_s - 1e-9:
            sim.step()
        (_, _, yaw), _, _ = sim.pose()
        oy = -BALL_OFFSET_ABS_Y
        sim.place_ball((BALL_OFFSET_X + rng.uniform(-BALL_NOISE, BALL_NOISE), oy + rng.uniform(-BALL_NOISE, BALL_NOISE)), yaw)
        sim.heading = yaw
        sim.policy.ort_session = sim.kick_session
        sim.t = 0.0
    if reset_last_action:
        sim.policy.last_action[:] = 0
    b0, _ = sim.ball()
    kdir = np.array([math.cos(sim.heading), math.sin(sim.heading)])
    peak, contact_t, v_after, fell_t, swapped = 0.0, None, None, None, False
    while sim.t < EPISODE_S - 1e-9:
        if handoff is not None and not swapped and sim.t >= handoff_at:
            sim.policy.ort_session = handoff
            swapped = True
        if push is not None and abs(sim.t - push[0]) < CONTROL_DT / 2:
            a = rng.uniform(0, 2 * math.pi)
            sim.push(push[1] * math.cos(a), push[1] * math.sin(a))
        sim.step()
        _, bv = sim.ball()
        fwd = float(bv @ kdir)
        peak = max(peak, fwd)
        if contact_t is None and np.linalg.norm(bv) > MOVED:
            contact_t = sim.t
        if contact_t is not None and v_after is None and sim.t >= contact_t + 0.25:
            v_after = fwd
        _, tilt, z = sim.pose()
        if fell_t is None and (tilt > FALL_TILT_DEG or z < FALL_Z):
            fell_t = sim.t
    b1, _ = sim.ball()
    disp = b1 - b0
    fwd_travel = float(disp @ kdir)
    lat = float(disp @ np.array([-kdir[1], kdir[0]]))
    _, tilt, z = sim.pose()
    up_end = tilt < UP_TILT_DEG and z > UP_Z
    if fell_t is not None:
        end = "fell-recovered" if up_end else "fell"
    else:  # never crossed training's fell_over, but may still not be standing (e.g. a held lean)
        end = "standing" if up_end else "not-upright"
    return {"contact_t": contact_t, "peak_fwd": round(peak, 3),
            "fwd_0.25s": None if v_after is None else round(v_after, 3),
            "travel_fwd": round(fwd_travel, 3), "travel_lat": round(lat, 3),
            "dir_err_deg": round(math.degrees(math.atan2(lat, fwd_travel)), 1) if np.hypot(fwd_travel, lat) > 0.02 else None,
            "fell_t": fell_t, "end": end, "end_tilt_deg": round(tilt, 1), "end_z_mm": round(z * 1000, 1)}


def pct(xs, q):
    return float(np.percentile(xs, q)) if xs else float("nan")


def summarize(rows, target, label):
    n = len(rows)
    kicked = [r for r in rows if r["contact_t"] is not None and r["peak_fwd"] > 0.05]
    pk = [r["peak_fwd"] for r in kicked]
    tr = [r["travel_fwd"] for r in kicked]
    de = [abs(r["dir_err_deg"]) for r in kicked if r["dir_err_deg"] is not None]
    ends = {k: sum(r["end"] == k for r in rows) for k in ("standing", "not-upright", "fell-recovered", "fell")}
    nu = [r["end_tilt_deg"] for r in rows if r["end"] == "not-upright"]
    good = sum(r in kicked and r["end"] == "standing" for r in rows)
    lines = [f"== {label}: {n} episodes, target {target} m/s",
             f"  kicked (ball moved forward): {len(kicked)}/{n}",
             f"  peak ball speed: median {pct(pk, 50):.2f} m/s (p10 {pct(pk, 10):.2f}, p90 {pct(pk, 90):.2f}) "
             f"= {pct(pk, 50) / target:.1f}x target",
             f"  ball travel (forward, 5 s): median {pct(tr, 50):.2f} m (p10 {pct(tr, 10):.2f}, p90 {pct(tr, 90):.2f})",
             f"  direction error |deg|: median {pct(de, 50):.1f} (p90 {pct(de, 90):.1f}, max {max(de) if de else float('nan'):.1f})",
             f"  contact time: median {pct([r['contact_t'] for r in kicked], 50):.2f} s after the swap-in",
             f"  duck at end: standing {ends['standing']}/{n}, not upright (never fell, held a lean"
             f"{', median end tilt %.0f deg' % pct(nu, 50) if nu else ''}) {ends['not-upright']}, "
             f"fell-recovered {ends['fell-recovered']}, fell {ends['fell']}",
             f"  GOOD KICKS (kicked AND standing at the end, never fell): {good}/{n}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--target", type=float, default=1.0, help="BALL_TARGET_SPEED the policy trained with")
    ap.add_argument("--scene", choices=list(SCENES), default="ball")
    ap.add_argument("--handoff", default=None, help="VelStand ONNX to swap back to")
    ap.add_argument("--handoff-at", type=float, default=2.0)
    ap.add_argument("--push-at", type=float, default=None)
    ap.add_argument("--push-v", type=float, default=0.3)
    ap.add_argument("--prestand", default=None, help="VelStand ONNX that holds the stand before the kick swap-in")
    ap.add_argument("--prestand-s", type=float, default=1.5)
    ap.add_argument("--reset-last-action", action="store_true", help="zero the last-action obs at the swap-in")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="csv path for per-episode rows")
    args = ap.parse_args()

    sim = KickSim(args.onnx, SCENES[args.scene])
    handoff = ort.InferenceSession(args.handoff) if args.handoff else None
    rng = np.random.default_rng(args.seed)
    push = (args.push_at, args.push_v) if args.push_at is not None else None
    t0 = time.time()
    prestand = ort.InferenceSession(args.prestand) if args.prestand else None
    rows = [run_episode(sim, rng, handoff, args.handoff_at, push, prestand, args.prestand_s, args.reset_last_action)
            for _ in range(args.episodes)]
    label = (f"{os.path.basename(args.onnx)} on {args.scene}"
             + (f", swapped in after {args.prestand_s}s of VelStand standing" if prestand else "")
             + (", last action zeroed at swap-in" if args.reset_last_action else "")
             + (f", VelStand handoff at {args.handoff_at}s" if handoff else ", kick policy for the full 5 s")
             + (f", push {args.push_v} m/s at {args.push_at}s" if push else ""))
    print(summarize(rows, args.target, label))
    print(f"  ({time.time() - t0:.0f} s wall)")
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()

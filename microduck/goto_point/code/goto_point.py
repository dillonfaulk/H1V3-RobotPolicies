"""Drive a VelStand policy to (x, y) waypoints with the v0 go-to-point supervisor.

Run from the microduck_rl repo root.

Live (MuJoCo viewer, keys typed in the terminal: P push, SPACE e-stop, Q quit):
  uv run python scripts/goto_point.py <onnx> --viewer --waypoints "1.2,0;1.2,1;0,0"

Headless acceptance suite (writes logs/goto/<stamp>_<suite>/):
  uv run python scripts/goto_point.py <onnx> --suite all

Sim settings match logs/headless_rehearsal.py's fixed harness: BAM actuator,
1-step action delay, training solver, all-collisions scene, projected gravity.
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import infer_policy as ip  # noqa: E402
from goto_supervisor import GotoConfig, GotoSupervisor  # noqa: E402

SCENE = "src/mjlab_microduck/robot/microduck/scene_allcollisions.xml"
CONTROL_DT = 0.02  # 50 Hz, physics 5 ms x decimation 4
PUSH_MAX = 1.0     # infer_policy's P key push


class DuckSim:
    """One CPU MuJoCo duck driven by the policy at 50 Hz."""

    def __init__(self, onnx, scene=SCENE, delay=1, train_solver=True):
        ip.MICRODUCK_XML = scene
        bam_model = ip.load_bam_model(ip.BAM_KP_FW, 7.4, 0.0)
        self.model, self.data, self.bam, _ = ip.load_mujoco_with_bam(scene, bam_model, 0.005, 0.1, ip.BAM_VIN_MIN)
        if train_solver:
            self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
            self.model.opt.iterations, self.model.opt.ls_iterations = 10, 20
        self.policy = ip.PolicyInference(self.model, self.data, bam_ctrl=self.bam, walking_onnx_path=onnx,
                                         new_cmd_obs=True, use_projected_gravity=True,
                                         delay_min_lag=delay, delay_max_lag=delay)
        fj = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        self.qa, self.va = self.model.jnt_qposadr[fj], self.model.jnt_dofadr[fj]
        self.trunk_body = self.model.jnt_bodyid[fj]
        self.t = 0.0

    def reset(self, x=0.0, y=0.0, yaw=0.0):
        d, qa = self.data, self.qa
        d.qpos[qa:qa + 3] = [x, y, 0.125]
        d.qpos[qa + 3:qa + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        for i, q in enumerate(self.policy.joint_qpos_indices):
            d.qpos[q] = self.policy.default_pose[i]
        d.qvel[:] = 0
        d.time = 0.0
        p = self.policy  # per-episode policy state: last action obs + delay buffer
        p.last_action[:] = 0
        if p.action_buffer is not None:
            for a in p.action_buffer:
                a[:] = 0
        self.bam.reset(d.qpos)
        self.policy.set_position_targets(self.policy.default_pose)
        self.set_twist((0.0, 0.0, 0.0))
        mujoco.mj_forward(self.model, d)
        self.t = 0.0

    def set_twist(self, twist):
        # Twist slots only; head/body pose slots stay zero, as in infer_policy driving.
        # (set_vel_cmd would print every tick.)
        self.policy.vel_cmd = np.array(twist, dtype=np.float32)
        self.policy._update_command()

    def step(self):
        self.policy.apply_action(self.policy.infer())
        for _ in range(4):
            self.bam.update()
            mujoco.mj_step(self.model, self.data)
        self.t += CONTROL_DT

    def pose(self):
        """((x, y, yaw), tilt_deg, trunk_z) from ground truth."""
        q = self.data.qpos
        w, x, y, z = q[self.qa + 3:self.qa + 7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))
        return (float(q[self.qa]), float(q[self.qa + 1]), yaw), tilt, float(q[self.qa + 2])

    def achieved(self):
        """Body-frame (vx, wz)."""
        quat = self.data.qpos[self.qa + 3:self.qa + 7].astype(np.float32)
        vb = self.policy.quat_rotate_inverse(quat, self.data.qvel[self.va:self.va + 3].astype(np.float32))
        return float(vb[0]), float(self.data.qvel[self.va + 5])

    def push(self, vx, vy):
        """Overwrite the trunk's world xy velocity, like the training push event / P key."""
        self.data.qvel[self.va], self.data.qvel[self.va + 1] = vx, vy


class RunLog:
    """Events (jsonl) + full 50 Hz command trace (csv) for one run."""

    def __init__(self, path=None, echo=True):
        self.events, self.trace, self.path, self.echo = [], [], path, echo

    def event(self, e):
        self.events.append(e)
        if self.echo and e["event"] != "GO_TO_TRACE":
            print(f"  [{e['t']:6.2f}s] {e['event']:12s} " +
                  " ".join(f"{k}={v}" for k, v in e.items() if k not in ("t", "event")), flush=True)

    def tick(self, sim, sup, twist):
        (x, y, yaw), tilt, z = sim.pose()
        avx, awz = sim.achieved()
        L = sup.last
        self.trace.append({"t": round(sim.t, 3), "x": round(x, 4), "y": round(y, 4), "yaw": round(yaw, 4),
                           "tilt_deg": round(tilt, 1), "trunk_z": round(z, 4), "state": L["state"],
                           "reason": L["reason"], "wp": L["wp"],
                           "dist": None if L["dist"] is None else round(L["dist"], 4),
                           "heading_err": None if L["err"] is None else round(L["err"], 4),
                           "cmd_vx": twist[0], "cmd_vy": twist[1], "cmd_wz": round(twist[2], 4),
                           "ach_vx": round(avx, 4), "ach_wz": round(awz, 4)})

    def save(self, stem):
        if not self.trace:
            return
        with open(stem + "_events.jsonl", "w") as f:
            for e in self.events:
                f.write(json.dumps(e) + "\n")
        with open(stem + "_trace.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.trace[0].keys()))
            w.writeheader()
            w.writerows(self.trace)


def leg_report(trace):
    """Commanded vs achieved per waypoint leg, split by arc/pivot and turn direction."""
    lines, legs = [], {}
    for r in trace:
        if r["reason"] not in ("arc", "pivot"):
            continue
        kind = r["reason"]
        if abs(r["cmd_wz"]) >= 0.05:
            kind += " left" if r["cmd_wz"] > 0 else " right"
        else:
            kind += " straight"
        legs.setdefault(r["wp"], {}).setdefault(kind, []).append(r)
    for wp, kinds in legs.items():
        for kind, rows in sorted(kinds.items()):
            c = np.array([(r["cmd_vx"], r["cmd_wz"], r["ach_vx"], r["ach_wz"]) for r in rows])
            lines.append(f"    wp {wp} {kind:15s} {len(rows) * CONTROL_DT:5.1f}s  cmd vx {c[:,0].mean():+.2f} "
                         f"wz {c[:,1].mean():+.2f}  ->  achieved vx {c[:,2].mean():+.2f} wz {c[:,3].mean():+.2f}")
    return lines


def run_route(sim, waypoints, arena, start, timeout_s, cfg=None, pushes=(), echo=True):
    """Run one route headless. pushes = [(trigger(sim, sup) -> bool, (vx, vy))], each fired once."""
    sim.reset(*start)
    log = RunLog(echo=echo)
    sup = GotoSupervisor(waypoints, arena, cfg, log=log.event)
    pending = list(pushes)
    marker_history = [tuple(i for i, _ in sup.queue)]
    while sim.t < timeout_s and sup.state != "DONE":
        sup.update(sim.t, *sim.pose())
        twist = sup.command(sim.t)
        sim.set_twist(twist)
        log.tick(sim, sup, twist)
        for p in list(pending):
            if p[0](sim, sup):
                sim.push(*p[1])
                log.event({"t": round(sim.t, 3), "event": "PUSH", "v": list(p[1])})
                pending.remove(p)
        sim.step()
        q = tuple(i for i, _ in sup.queue)
        if q != marker_history[-1]:
            marker_history.append(q)
    for _ in range(50):  # 1 s of standing after the route, so a stop-time fall would show
        sup.update(sim.t, *sim.pose())
        twist = sup.command(sim.t)
        sim.set_twist(twist)
        log.tick(sim, sup, twist)
        sim.step()
    return sup, log, marker_history


# ---------------------------------------------------------------- acceptance suites

ARENA = (-2.5, 2.5, -2.5, 2.5)


def unrecovered(sup, log):
    """Falls with no matching RECOVERED (i.e. still down when the run ended)."""
    count = lambda name: sum(e["event"] == name for e in log.events)  # noqa: E731
    return count("FALL") - count("RECOVERED")


def suite_single(sim, out, trials=10, seed=0):
    rng = np.random.default_rng(seed)
    print(f"\n== single point: {trials} trials, random start pose + heading, goal 0.6-2.0 m any bearing ==")
    ok = 0
    for k in range(trials):
        start = (rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-math.pi, math.pi))
        d, b = rng.uniform(0.6, 2.0), rng.uniform(-math.pi, math.pi)  # bearing relative to start heading
        goal = (float(np.clip(start[0] + d * math.cos(start[2] + b), -2.3, 2.3)),
                float(np.clip(start[1] + d * math.sin(start[2] + b), -2.3, 2.3)))
        dist = math.hypot(goal[0] - start[0], goal[1] - start[1])
        sup, log, _ = run_route(sim, [goal], ARENA, start, timeout_s=10 + 2 * dist / 0.2, echo=False)
        arrived = sup.arrived == [1]
        t_arr = next((e["t"] for e in log.events if e["event"] == "GO_TO_ARRIVE"), None)
        bad = unrecovered(sup, log)
        ok += arrived and bad == 0
        print(f"  trial {k}: start ({start[0]:+.2f},{start[1]:+.2f}, {math.degrees(start[2]):+4.0f}deg) "
              f"goal ({goal[0]:+.2f},{goal[1]:+.2f}) dist {dist:.2f} m, bearing {math.degrees(b):+4.0f}deg -> "
              f"{'ARRIVED %.1fs' % t_arr if arrived else 'NOT ARRIVED'}  falls {sup.falls}"
              f"{' UNRECOVERED' if bad else ''}", flush=True)
        log.save(f"{out}/single_{k}")
    print(f"  single point: {ok}/{trials} arrivals with no unrecovered falls")
    return ok == trials


CHAIN = [(1.2, 0.0), (1.2, 1.0), (2.2, 1.3), (0.0, 0.0)]  # straight, left 90, right arc, behind (pivot)


def suite_chain(sim, out):
    print(f"\n== chain: {CHAIN} from (0,0) facing +x ==")
    sup, log, markers = run_route(sim, CHAIN, ARENA, (0, 0, 0), timeout_s=90)
    log.save(f"{out}/chain")
    print("  marker queue over time: " + " -> ".join(str(list(m)) for m in markers))
    print("\n".join(leg_report(log.trace)))
    ok = sup.arrived == [1, 2, 3, 4] and unrecovered(sup, log) == 0
    print(f"  chain: arrived {sup.arrived} in {sim.t:.1f}s, falls {sup.falls} -> {'PASS' if ok else 'FAIL'}")
    return ok


def suite_fall(sim, out, push_v=1.6, retries=4):
    """Knock it over mid-leg; retry with a harder push if the first one doesn't topple it."""
    print(f"\n== mid-route fall: route (0,0) -> (2.0, 0) -> (2.0, 1.0), lateral push at x > 0.8 m ==")
    for k in range(retries):
        v = push_v + 0.4 * k
        trig = (lambda s, sup: s.pose()[0][0] > 0.8 and sup.current is not None and sup.current[0] == 1)
        sup, log, _ = run_route(sim, [(2.0, 0.0), (2.0, 1.0)], ARENA, (0, 0, 0), timeout_s=60,
                                pushes=[(trig, (0.0, v))], echo=False)
        if sup.falls:
            break
        print(f"  push {v:.1f} m/s did not topple it, retrying harder", flush=True)
    for e in log.events:
        if e["event"] not in ("GO_TO_TRACE",):
            print(f"  [{e['t']:6.2f}s] {e['event']:12s} " +
                  " ".join(f"{k}={v}" for k, v in e.items() if k not in ("t", "event")))
    log.save(f"{out}/fall")
    fall_t = next((e["t"] for e in log.events if e["event"] == "FALL"), None)
    rec_t = next((e["t"] for e in log.events if e["event"] == "RECOVERED"), None)
    zero_during = all(r["cmd_vx"] == 0 and r["cmd_wz"] == 0 for r in log.trace
                      if fall_t is not None and rec_t is not None and fall_t <= r["t"] < rec_t)
    ok = bool(sup.falls) and sup.arrived == [1, 2] and unrecovered(sup, log) == 0 and zero_during
    print(f"  fall: falls {sup.falls}, zero twist while down {zero_during}, "
          f"FALL->RECOVERED {None if fall_t is None or rec_t is None else round(rec_t - fall_t, 2)}s, "
          f"arrived {sup.arrived} -> {'PASS' if ok else 'FAIL'}")
    return ok


def suite_geofence(sim, out):
    wps = [(1.0, 0.0), (4.0, 0.0), (1.0, 1.0)]
    print(f"\n== geofence: arena x/y in [-2.5, 2.5], waypoints {wps} (#2 is outside) ==")
    sup, log, _ = run_route(sim, wps, ARENA, (0, 0, 0), timeout_s=40)
    log.save(f"{out}/geofence")
    rej = [e for e in log.events if e["event"] == "GO_TO_REJECT"]
    first_start = next(i for i, e in enumerate(log.events) if e["event"] == "GO_TO_START")
    before_motion = all(log.events.index(e) < first_start for e in rej) and all(e["t"] == 0 for e in rej)
    ok = [e["wp"] for e in rej] == [2] and before_motion and sup.arrived == [1, 3]
    print(f"  geofence: rejected {[e['wp'] for e in rej]} before motion {before_motion}, arrived {sup.arrived}"
          f" -> {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------- live viewer

def draw_scene(scn, sup, arena, sim):
    scn.ngeom = 0

    def add(gtype, size, pos, rgba, mat=np.eye(3)):
        if scn.ngeom >= scn.maxgeom:
            return None
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float),
                            np.asarray(mat, float).flatten(), np.asarray(rgba, np.float32))
        scn.ngeom += 1
        return g

    def line(a, b, rgba, width=0.006):
        g = add(mujoco.mjtGeom.mjGEOM_CAPSULE, [1, 1, 1], [0, 0, 0], rgba)
        if g is not None:
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a, float), np.asarray(b, float))

    xmin, xmax, ymin, ymax = arena
    corners = [(xmin, ymin, 0.002), (xmax, ymin, 0.002), (xmax, ymax, 0.002), (xmin, ymax, 0.002)]
    for i in range(4):
        line(corners[i], corners[(i + 1) % 4], [1.0, 0.2, 0.2, 0.8], 0.008)
    r = sup.cfg.arrival_radius
    for n, (wid, (x, y)) in enumerate(sup.queue):  # arrived waypoints have left the queue -> despawned
        rgba = [0.1, 0.9, 0.2, 0.55] if n == 0 else [1.0, 0.6, 0.1, 0.45]
        add(mujoco.mjtGeom.mjGEOM_CYLINDER, [r, 0.003, 0], [x, y, 0.003], rgba)
        add(mujoco.mjtGeom.mjGEOM_CYLINDER, [0.006, 0.08, 0], [x, y, 0.08], rgba[:3] + [0.9])
    if sup.current is not None:
        (px, py, _), _, _ = sim.pose()
        line((px, py, 0.01), (*sup.current[1], 0.01), [0.1, 0.9, 0.2, 0.6], 0.003)


def run_viewer(sim, waypoints, arena, cfg):
    import mujoco.viewer
    sim.reset(0, 0, 0)
    log = RunLog()
    sup = GotoSupervisor(waypoints, arena, cfg, log=log.event)
    import random
    print("\nKeys (type in THIS terminal): P = random push, SPACE = e-stop on/off, Q = quit\n")
    with ip.TerminalInput() as term, \
            mujoco.viewer.launch_passive(sim.model, sim.data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = sim.trunk_body
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 1.6, -35, 135
        quit_req = False
        while viewer.is_running() and not quit_req:
            t0 = time.time()
            for key in term.get_keys():
                if key == "q":
                    quit_req = True
                elif key == " ":
                    sup.estop = not sup.estop
                    log.event({"t": round(sim.t, 3), "event": "ESTOP_ON" if sup.estop else "ESTOP_OFF"})
                elif key == "p":
                    a = random.uniform(0, 2 * math.pi)
                    sim.push(PUSH_MAX * math.cos(a), PUSH_MAX * math.sin(a))
                    log.event({"t": round(sim.t, 3), "event": "PUSH", "angle_deg": round(math.degrees(a))})
            sup.update(sim.t, *sim.pose())
            twist = sup.command(sim.t)
            sim.set_twist(twist)
            log.tick(sim, sup, twist)
            L = sup.last
            if int(sim.t / CONTROL_DT) % 50 == 0 and L["dist"] is not None:
                avx, awz = sim.achieved()
                print(f"  wp {L['wp']} {L['reason']:6s} dist {L['dist']:.2f} m  err {math.degrees(L['err']):+5.0f}deg  "
                      f"cmd vx {twist[0]:.2f} wz {twist[2]:+.2f}  achieved vx {avx:+.2f} wz {awz:+.2f}", flush=True)
            sim.step()
            draw_scene(viewer.user_scn, sup, arena, sim)
            viewer.sync()
            time.sleep(max(0.0, CONTROL_DT - (time.time() - t0)))
    out = f"logs/goto/{time.strftime('%Y-%m-%d_%H-%M-%S')}_viewer"
    os.makedirs(out, exist_ok=True)
    log.save(f"{out}/route")
    print(f"\narrived {sup.arrived}, rejected {sup.rejected}, falls {sup.falls}; log: {out}/")
    print("\n".join(leg_report(log.trace)))


def parse_points(s):
    return [tuple(float(v) for v in p.split(",")) for p in s.split(";") if p.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--waypoints", default="1.2,0;1.2,1;2.2,1.3;0,0", help='"x1,y1;x2,y2;..." world frame, m')
    ap.add_argument("--arena", default="-2.5,2.5,-2.5,2.5", help="xmin,xmax,ymin,ymax geofence, m")
    ap.add_argument("--suite", choices=["single", "chain", "fall", "geofence", "all"], default="all")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--delay", type=int, default=1)
    ap.add_argument("--scene", default=SCENE)
    args = ap.parse_args()

    sim = DuckSim(args.onnx, scene=args.scene, delay=args.delay)
    if args.viewer:
        run_viewer(sim, parse_points(args.waypoints), tuple(float(v) for v in args.arena.split(",")), GotoConfig())
        return
    out = f"logs/goto/{time.strftime('%Y-%m-%d_%H-%M-%S')}_{args.suite}"
    os.makedirs(out, exist_ok=True)
    print(f"onnx={args.onnx}\nlogs -> {out}/")
    results = {}
    if args.suite in ("single", "all"):
        results["single"] = suite_single(sim, out, args.trials)
    if args.suite in ("chain", "all"):
        results["chain"] = suite_chain(sim, out)
    if args.suite in ("fall", "all"):
        results["fall"] = suite_fall(sim, out)
    if args.suite in ("geofence", "all"):
        results["geofence"] = suite_geofence(sim, out)
    print("\n== summary: " + ", ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in results.items()))


if __name__ == "__main__":
    main()

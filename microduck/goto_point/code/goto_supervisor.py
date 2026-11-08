"""Go-to-point v0 supervisor for the Microduck VelStand policy (zero training).

A deterministic FSM that sits OUTSIDE the policy and only writes the twist
command (vx, vy, wz). It knows nothing about MuJoCo: the caller feeds it a pose
each control tick and applies the twist it returns. Ported from the asimov-1 v0
executor, with the duck's measured round-3 command response designed in:

- Walking speed is a step, not a ramp (0.10 m/s is dead, ~0.13 walks), so the
  supervisor never commands a creep: vx is either 0 or CRUISE_VX. Arrival is
  "cut to zero inside the radius", never a proportional taper.
- Turning in place is dead at <= 0.4 rad/s but turning WHILE walking tracks
  down to ~0.1 rad/s (measured, CPU sim, vx 0.20). So steering is a forward arc
  with a proportional yaw rate, and pivots (vx 0) use PIVOT_WZ = 0.8 rad/s.
- Strafe is weak and backing curves, so paths are forward-only with vy = 0.
- Zero twist means "stand" to this policy, and VelStand gets up on its own, so
  the whole fall response is: command zero until upright and settled.

Pose source is ground truth from the sim in v0; on hardware the same update()
call would take the onboard odometry pose instead.
"""
import math
from dataclasses import dataclass

# Hard command clamps, applied after the FSM. These are the audit numbers.
VX_MIN, VX_MAX = 0.0, 0.25   # forward only
WZ_ABS_MAX = 1.0


@dataclass
class GotoConfig:
    arrival_radius: float = 0.15   # m
    cruise_vx: float = 0.20        # m/s, above the ~0.13 walk-onset step
    k_w: float = 1.5               # arc yaw gain, rad/s per rad of heading error
    arc_wz_max: float = 1.0        # rad/s
    pivot_wz: float = 0.8          # rad/s, in-place turns are dead <= 0.4
    pivot_enter: float = math.radians(60)  # heading error that starts a pivot
    pivot_exit: float = math.radians(20)   # heading error that ends it (hysteresis)
    orbit_margin: float = 1.3      # left arcs achieve only ~80% of the commanded yaw rate
    fall_tilt_deg: float = 35.0    # VelStand's "fallen" BC threshold
    fall_z: float = 0.06           # m, trunk height
    up_tilt_deg: float = 25.0      # VelStand RECOVERED_UP_TILT_DEG
    up_z: float = 0.09             # VelStand RECOVERED_UP_Z
    settle_s: float = 1.0          # upright this long before navigating again
    watchdog_s: float = 0.2        # pose older than this -> zero twist
    trace_period_s: float = 1.0    # throttled GO_TO_TRACE events


def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def clamp_twist(vx, vy, wz):
    """Final clamp, the one place the command limits live."""
    return (min(max(vx, VX_MIN), VX_MAX), 0.0, min(max(wz, -WZ_ABS_MAX), WZ_ABS_MAX))


def goal_in_base(pose, goal):
    """World-frame goal -> (x_local, y_local) in the robot's yaw frame."""
    x, y, yaw = pose
    dx, dy = goal[0] - x, goal[1] - y
    c, s = math.cos(yaw), math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


class GotoSupervisor:
    """Waypoint queue + orient/arc FSM + fall rule + geofence + watchdog.

    States: NAVIGATE (mode ARC or PIVOT), RECOVERING, ESTOP, DONE.
    `log(event_dict)` receives every event; events carry the sim time `t`.
    """

    def __init__(self, waypoints, arena, cfg=None, log=None):
        self.cfg = cfg or GotoConfig()
        self.arena = arena  # (xmin, xmax, ymin, ymax), world frame
        self._log = log or (lambda e: None)
        self.queue = []     # [(id, (x, y))] still to visit, in order
        self.arrived = []   # ids in arrival order
        self.rejected = []
        self.state, self.mode = "DONE", "ARC"
        self.estop = False
        self._pose_t = None
        self._up_since = None
        self._last_trace_t = -math.inf
        self._started_id = None
        self._watchdog_tripped = False
        self.falls = 0
        self.last = {}      # last tick's internals, for traces and markers
        for i, wp in enumerate(waypoints, start=1):
            if self._inside(wp):
                self.queue.append((i, (float(wp[0]), float(wp[1]))))
            else:
                self.rejected.append(i)
                self._emit(0.0, "GO_TO_REJECT", wp=i, goal=list(wp),
                           reason=f"outside arena x[{arena[0]}, {arena[1]}] y[{arena[2]}, {arena[3]}]")
        if self.queue:
            self.state = "NAVIGATE"

    def _inside(self, wp):
        xmin, xmax, ymin, ymax = self.arena
        return xmin <= wp[0] <= xmax and ymin <= wp[1] <= ymax

    def _emit(self, t, event, **kw):
        self._log({"t": round(t, 3), "event": event, **kw})

    @property
    def current(self):
        return self.queue[0] if self.queue else None

    def update(self, t, pose, tilt_deg, trunk_z):
        """Feed one pose sample (sim/wall time t, pose = (x, y, yaw))."""
        self._pose_t = t
        self._pose = pose
        self._tilt, self._z = tilt_deg, trunk_z

    def command(self, t):
        """Twist for this tick. Call after update(); returns the clamped (vx, vy, wz)."""
        cfg = self.cfg
        if self._pose_t is None or t - self._pose_t > cfg.watchdog_s:
            if not self._watchdog_tripped:
                self._watchdog_tripped = True
                self._emit(t, "WATCHDOG", stale_s=None if self._pose_t is None else round(t - self._pose_t, 3))
            return self._out(t, (0.0, 0.0, 0.0), reason="watchdog")
        self._watchdog_tripped = False
        if self.estop:
            return self._out(t, (0.0, 0.0, 0.0), reason="estop")
        if self.state == "DONE":
            return self._out(t, (0.0, 0.0, 0.0), reason="done")

        wp_id = self.current[0]
        fallen = self._tilt > cfg.fall_tilt_deg or self._z < cfg.fall_z
        if self.state == "NAVIGATE" and fallen:
            self.state, self._up_since = "RECOVERING", None
            self.falls += 1
            self._emit(t, "FALL", wp=wp_id, tilt_deg=round(self._tilt, 1), trunk_z=round(self._z, 3),
                       pose=[round(v, 3) for v in self._pose])
        if self.state == "RECOVERING":
            up = self._tilt < cfg.up_tilt_deg and self._z > cfg.up_z
            self._up_since = (self._up_since if self._up_since is not None else t) if up else None
            if self._up_since is None or t - self._up_since < cfg.settle_s:
                return self._out(t, (0.0, 0.0, 0.0), reason="recovering")
            self.state, self.mode = "NAVIGATE", "ARC"
            self._emit(t, "RECOVERED", wp=wp_id, pose=[round(v, 3) for v in self._pose])

        if self._started_id != wp_id:
            self._started_id = wp_id
            self._emit(t, "GO_TO_START", wp=wp_id, goal=list(self.current[1]),
                       pose=[round(v, 3) for v in self._pose])
        xl, yl = goal_in_base(self._pose, self.current[1])
        dist, err = math.hypot(xl, yl), math.atan2(yl, xl)
        if dist < cfg.arrival_radius:
            self.arrived.append(wp_id)
            self.queue.pop(0)
            self._emit(t, "GO_TO_ARRIVE", wp=wp_id, dist=round(dist, 3), pose=[round(v, 3) for v in self._pose])
            if not self.queue:
                self.state = "DONE"
                self._emit(t, "GO_TO_DONE", arrived=self.arrived, rejected=self.rejected, falls=self.falls)
            self.mode = "ARC"
            return self._out(t, (0.0, 0.0, 0.0), reason="arrived", dist=dist, err=err, goal_local=(xl, yl))

        # A forward arc at cruise speed with the max yaw rate has radius
        # r = cruise_vx / arc_wz_max; a goal off to the side closer than
        # 2 r sin|err| sits inside that circle and the arc would orbit it.
        r_min = cfg.orbit_margin * cfg.cruise_vx / cfg.arc_wz_max
        orbit = dist < 2 * r_min * abs(math.sin(err)) and abs(err) > cfg.pivot_exit
        if self.mode == "ARC" and (abs(err) > cfg.pivot_enter or orbit):
            self.mode = "PIVOT"
        elif self.mode == "PIVOT" and abs(err) < cfg.pivot_exit:
            self.mode = "ARC"
        if self.mode == "PIVOT":
            twist = (0.0, 0.0, math.copysign(cfg.pivot_wz, err))
        else:
            twist = (cfg.cruise_vx, 0.0, max(-cfg.arc_wz_max, min(cfg.arc_wz_max, cfg.k_w * err)))
        return self._out(t, twist, reason=self.mode.lower(), dist=dist, err=err, goal_local=(xl, yl))

    def _out(self, t, twist, reason, dist=None, err=None, goal_local=None):
        twist = clamp_twist(*twist)
        self.last = {"t": t, "state": self.state, "reason": reason, "wp": self.current[0] if self.current else None,
                     "dist": dist, "err": err, "goal_local": goal_local, "cmd": twist}
        if t - self._last_trace_t >= self.cfg.trace_period_s:
            self._last_trace_t = t
            self._emit(t, "GO_TO_TRACE", wp=self.last["wp"], state=self.state, reason=reason,
                       goal_local=None if goal_local is None else [round(v, 3) for v in goal_local],
                       heading_err_deg=None if err is None else round(math.degrees(err), 1),
                       dist=None if dist is None else round(dist, 3),
                       cmd=[round(v, 3) for v in twist])
        return twist

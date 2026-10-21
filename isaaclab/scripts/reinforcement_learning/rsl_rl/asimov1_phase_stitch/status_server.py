"""Slow-cadence status/control HTTP server for the phase machine -- stdlib `http.server`, no new
dependency. Runs in a background thread so the 60Hz main loop only ever touches `PhaseState`
through a lock, cheaply, once per tick -- never blocks on HTTP I/O.

This is deliberately NOT the hot path: h1v3-runtime's `robot_supervisor` agent tools poll/control
this endpoint at ordinary chat-agent cadence (occasional, human-triggered), never per physics tick.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class PhaseState:
    def __init__(self):
        self._lock = threading.Lock()
        self.phase = "WALK"
        self.phase_since_t = time.time()
        self.last_gate_values: dict = {}
        self.last_transition: str | None = None
        self.walk_success_count = 0
        self.fall_count = 0
        self.push_requested = False
        self.reset_requested = False
        self.goal_queue: list[tuple[float, float]] = []
        self.current_goal: tuple[float, float] | None = None
        self.goto_arrivals = 0
        self._goto_last_vals: dict = {}

    def update(self, phase: str, gate_values: dict, sim_time: float = None) -> None:
        with self._lock:
            if phase != self.phase:
                self.last_transition = f"{self.phase} -> {phase}"
                t_str = f"t={sim_time:.1f}s " if sim_time is not None else ""
                print(f"[phase_stitch] {t_str}{self.last_transition}  gates={gate_values}")
                self.phase = phase
                self.phase_since_t = time.time()
            self.last_gate_values = gate_values

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "phase": self.phase,
                "time_in_phase_s": round(time.time() - self.phase_since_t, 2),
                "last_gate_values": self.last_gate_values,
                "last_transition": self.last_transition,
                "walk_success_count": self.walk_success_count,
                "fall_count": self.fall_count,
                "goto": {
                    "current_goal": self.current_goal,
                    "queue_remaining": len(self.goal_queue),
                    "arrivals": self.goto_arrivals,
                    **self._goto_last_vals,
                },
            }

    def request_push(self) -> None:
        with self._lock:
            self.push_requested = True

    def request_reset(self) -> None:
        with self._lock:
            self.reset_requested = True

    def drain_requests(self) -> tuple[bool, bool]:
        """Called once per tick by the main loop -- atomically reads and clears both flags."""
        with self._lock:
            push, reset = self.push_requested, self.reset_requested
            self.push_requested = False
            self.reset_requested = False
            return push, reset

    def set_goals(self, waypoints: list[tuple[float, float]]) -> None:
        """Replaces the waypoint queue and makes the first one current immediately. An empty list
        clears navigation entirely, falling the executor back to its original fixed-command demo
        behavior -- this is the v0 go-to-point queue (goto.py), a persistent state field (not a
        one-shot request like push/reset above), since the main loop needs to read it every tick."""
        with self._lock:
            self.goal_queue = list(waypoints)
            self.current_goal = self.goal_queue.pop(0) if self.goal_queue else None

    def get_current_goal(self) -> tuple[float, float] | None:
        with self._lock:
            return self.current_goal

    def get_queue_snapshot(self) -> tuple[tuple[float, float] | None, list[tuple[float, float]]]:
        """For the waypoint visualizer -- the current goal plus the still-queued ones (not consumed)."""
        with self._lock:
            return self.current_goal, list(self.goal_queue)

    def advance_goal(self) -> tuple[float, float] | None:
        """Called by the main loop once an arrival has been held for goto.ARRIVAL_HOLD_TICKS --
        pops the next queued waypoint, or clears current_goal (back to the default demo command) if
        the queue is empty."""
        with self._lock:
            self.goto_arrivals += 1
            self.current_goal = self.goal_queue.pop(0) if self.goal_queue else None
            return self.current_goal

    def update_goto(self, vals: dict) -> None:
        with self._lock:
            self._goto_last_vals = vals


def _make_handler(state: PhaseState):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/status":
                self._send_json(state.snapshot())
            else:
                self._send_json({"error": "not found"}, 404)

        def do_POST(self):
            if self.path == "/control/push":
                state.request_push()
                self._send_json({"ok": True, "queued": "push"})
            elif self.path == "/control/reset":
                state.request_reset()
                self._send_json({"ok": True, "queued": "reset"})
            elif self.path == "/control/goto":
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    self._send_json({"error": "invalid JSON body"}, 400)
                    return
                waypoints = payload.get("waypoints")
                if waypoints is None and "x" in payload and "y" in payload:
                    waypoints = [[payload["x"], payload["y"]]]
                if waypoints is None:
                    self._send_json(
                        {"error": "body must be {'x':.., 'y':..} or {'waypoints': [[x,y], ...]}"}, 400
                    )
                    return
                try:
                    parsed = [(float(p[0]), float(p[1])) for p in waypoints]
                except (TypeError, ValueError, IndexError):
                    self._send_json({"error": "waypoints must be a list of [x, y] pairs"}, 400)
                    return
                state.set_goals(parsed)
                self._send_json({"ok": True, "queued_waypoints": len(parsed)})
            else:
                self._send_json({"error": "not found"}, 404)

        def log_message(self, fmt, *args):
            pass  # silence stdlib's default per-request stderr logging

    return Handler


def start_status_server(state: PhaseState, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("localhost", port), _make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[phase_stitch] status server listening on http://localhost:{port}")
    return server

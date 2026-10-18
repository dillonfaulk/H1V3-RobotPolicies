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

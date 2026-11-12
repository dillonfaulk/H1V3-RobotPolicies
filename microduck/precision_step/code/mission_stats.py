"""Summarize kick_mission.py headless outputs: kicks, bumps, roll distance, kick direction, placement.

usage: python scripts/mission_stats.py <mission_s0.txt> [<mission_s1.txt> ...]
Reads the per-run "run k: ..." lines that run_headless prints.
"""
import re
import sys

import numpy as np


def parse(path):
    rows = []
    for line in open(path):
        m = re.match(r"\s+run (\d+): ball ([\d.]+) m away, kick heading\s*([+-]?\d+)deg -> (.*)", line)
        if not m:
            continue
        r = {"run": int(m[1]), "ball_dist": float(m[2]), "kh_deg": float(m[3])}
        body = re.sub(r"np\.float64\(([^)]*)\)", r"\1", m[4])
        for k, v in re.findall(r"(\w+)=(\[[^\]]*\]|[^,]+)", body):
            try:
                r[k] = eval(v, {}, {})  # numbers, lists, True/False/None
            except Exception:
                r[k] = v
        rows.append(r)
    return rows


def q(a):
    a = np.asarray([v for v in a if v is not None], float)
    if not len(a):
        return "n/a"
    return f"median {np.median(a):+.2f}, mean {a.mean():+.2f}, range [{a.min():+.2f}, {a.max():+.2f}], n={len(a)}"


def main(paths):
    rows = [r for p in paths for r in parse(p)]
    n = len(rows)
    kicked = [r for r in rows if (r.get("peak_ball_fwd") or 0) > 0.05]
    print(f"runs {n}: kicked {len(kicked)}/{n}, bumped {sum(r.get('ball_disturbed', 0) > 0 for r in rows)}, "
          f"falls {sum(r.get('falls', 0) for r in rows)}, aligned {sum(bool(r.get('aligned')) for r in rows)}, "
          f"modes {sorted({r.get('approach_mode', 'pulse') for r in rows})}")
    print("ball_travel m      :", q([r.get("ball_travel") for r in kicked]))
    print("peak_ball_fwd m/s  :", q([r.get("peak_ball_fwd") for r in kicked]))
    print("kick_dir_err deg   :", q([r.get("kick_dir_err_deg") for r in kicked]), "(+ = left of intended)")
    print("align_ex mm (foot) :", q([r.get("align_ex_mm") for r in rows]), "(+ = ball farther than trained)")
    print("align_ey mm (foot) :", q([r.get("align_ey_mm") for r in rows]))
    print("rfoot x vs HOME mm :", q([r["foot_vs_home_mm"][0] for r in rows if "foot_vs_home_mm" in r]))
    print("lfoot x vs HOME mm :", q([r["lfoot_vs_home_mm"][0] for r in rows if "lfoot_vs_home_mm" in r]))
    print("leg_dev deg        :", q([r.get("leg_dev_deg") for r in rows]))
    print("t_done s           :", q([r.get("t_done") for r in rows]))
    de = np.array([r["kick_dir_err_deg"] for r in kicked if r.get("kick_dir_err_deg") is not None])
    if len(de) > 4:
        ks = [r for r in kicked if r.get("kick_dir_err_deg") is not None]
        for name, f in (("align_ex_mm", lambda r: r.get("align_ex_mm")), ("align_ey_mm", lambda r: r.get("align_ey_mm")),
                        ("align_eh_deg", lambda r: r.get("align_eh_deg")), ("ball_dist", lambda r: r["ball_dist"]),
                        ("peak_ball_fwd", lambda r: r.get("peak_ball_fwd"))):
            x = np.array([f(r) for r in ks], float)
            print(f"  corr(kick_dir_err, {name:13s}) = {np.corrcoef(x, de)[0, 1]:+.2f}")
    if "--rows" in sys.argv:
        for r in rows:
            print(f"  {r['run']:2d} travel {r.get('ball_travel')}  dir {r.get('kick_dir_err_deg')}  "
                  f"ex {r.get('align_ex_mm')} ey {r.get('align_ey_mm')} eh {r.get('align_eh_deg')}")


if __name__ == "__main__":
    main([p for p in sys.argv[1:] if not p.startswith("--")])

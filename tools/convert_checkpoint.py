#!/usr/bin/env python3
"""
Convert an RSL-RL .pt checkpoint into safetensors + a JSON metadata sidecar.

Why: .pt files are pickle under the hood — loading one can execute arbitrary
code, so nobody should download a stranger's .pt off GitHub. Safetensors stores
only raw tensors (no code), loads faster, and is the standard for public model
sharing.

Usage:
    pip install safetensors            # one-time
    python convert_checkpoint.py model_14990.pt --out ./vault_drop \
        --robot asimov-1 --isaac-lab v3.0.0-beta2.patch1

Outputs in --out:
    actor.safetensors      deployable actor weights (policy network only)
    full.safetensors       every tensor found in the checkpoint (actor + critic + misc)
    metadata.json          everything that is NOT a tensor: iteration, shapes,
                           source hash, and REQUIRED manual fields you must fill in
                           (obs normalization stats — without these the policy
                           is a paperweight).

Run this on the Tower, where the checkpoints live. Commit the outputs to
H1V3-RobotPolicies next to the original .pt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_checkpoint(ckpt: dict) -> tuple[dict, dict, dict]:
    """Split an RSL-RL checkpoint into actor / critic / other tensors.

    Two save formats exist in the wild:
      - vanilla rsl_rl: one combined "model_state_dict" with "actor."/"critic." key prefixes.
      - this project's rsl_rl fork (used for every asimov-1/v1x1 checkpoint): separate
        top-level "actor_state_dict"/"critic_state_dict" dicts, plus "discriminator_state_dict"
        for AMP runs. Keys inside each (e.g. "mlp.0.weight") have no actor/critic prefix --
        the split is already done for us by which top-level dict they're in.
    """
    import torch  # local import: this script runs on the Tower, not here

    def tensors_only(d: dict) -> dict:
        return {k: v for k, v in d.items() if isinstance(v, torch.Tensor)}

    if "model_state_dict" in ckpt:
        actor, critic, other = {}, {}, {}
        for k, v in ckpt["model_state_dict"].items():
            if not isinstance(v, torch.Tensor):
                continue
            name = k.lower()
            if name.startswith("actor"):
                actor[k] = v
            elif name.startswith("critic"):
                critic[k] = v
            else:
                other[k] = v
        return actor, critic, other

    if "actor_state_dict" in ckpt and "critic_state_dict" in ckpt:
        actor = tensors_only(ckpt["actor_state_dict"])
        critic = tensors_only(ckpt["critic_state_dict"])
        other = {}
        for extra_key in ("discriminator_state_dict",):
            if extra_key in ckpt:
                prefix = extra_key.removesuffix("_state_dict")
                for k, v in tensors_only(ckpt[extra_key]).items():
                    other[f"{prefix}.{k}"] = v
        return actor, critic, other

    raise SystemExit(
        f"unrecognized checkpoint format (top-level keys: {list(ckpt.keys())}) -- "
        "neither 'model_state_dict' nor 'actor_state_dict'/'critic_state_dict' found."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint", type=Path, help="input .pt checkpoint (your own file)")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--robot", default="", help="robot name, e.g. asimov-1")
    ap.add_argument("--isaac-lab", default="", help="Isaac Lab version pin, e.g. v3.0.0-beta2.patch1")
    ap.add_argument("--run", default="", help="training run id / folder name for lineage")
    args = ap.parse_args()

    from safetensors.torch import save_file  # import here so --help works without it
    import torch

    ckpt_path = args.checkpoint
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    # NOTE: weights_only=False because RSL-RL checkpoints are dicts with
    # optimizer state and metadata. This is safe ONLY for your own files —
    # never torch.load an untrusted .pt this way. (That's the whole point
    # of this script.)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise SystemExit(f"{ckpt_path} doesn't look like an RSL-RL checkpoint (type: {type(ckpt)})")

    actor, critic, other = split_checkpoint(ckpt)
    if not actor:
        raise SystemExit("no actor tensors found — is this an RSL-RL actor-critic checkpoint?")

    save_file(actor, out / "actor.safetensors")
    save_file({**actor, **critic, **other}, out / "full.safetensors")

    def shape_map(d: dict) -> dict:
        return {k: list(v.shape) for k, v in d.items()}

    metadata = {
        "robot": args.robot,
        "isaac_lab_version": args.isaac_lab,
        "training_run": args.run,
        "source_file": ckpt_path.name,
        "source_sha256": sha256_of(ckpt_path),
        "converted_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "iteration": ckpt.get("iter"),
        "epoch": ckpt.get("epoch"),
        "actor_tensor_shapes": shape_map(actor),
        "critic_tensor_shapes": shape_map(critic),
        "other_tensor_shapes": shape_map(other),
        "non_tensor_keys": sorted(k for k in ckpt.keys() if k != "model_state_dict"),
        # ------------------------------------------------------------------
        # REQUIRED: fill these in before publishing. The checkpoint is useless
        # without the observation normalization it was trained with.
        # ------------------------------------------------------------------
        "obs_dim": None,               # e.g. 81
        "action_dim": None,            # e.g. 23
        "obs_normalization": {
            "type": None,              # e.g. "empirical" / "fixed" / "none"
            "mean": None,              # list of floats, length obs_dim
            "var": None,               # list of floats, length obs_dim
            "source": "REQUIRED — pull from your training env's normalizer",
        },
        "command_ranges": None,        # e.g. {"lin_vel_x": [-1.0, 1.0], ...}
        "control_hz": None,            # e.g. 60
        "action_scale": None,          # e.g. 0.5 (target = default + raw * scale)
        "reward_shape_lineage": None,  # e.g. "run 4 of asimov1_velocity_amp lineage"
        "notes": "",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"wrote {out}/")
    print(f"  actor.safetensors  ({len(actor)} tensors)")
    print(f"  full.safetensors   ({len(actor) + len(critic) + len(other)} tensors)")
    print(f"  metadata.json      iteration={metadata['iteration']}")
    print()
    print("NEXT: fill in metadata.json — especially obs_normalization — before publishing.")


if __name__ == "__main__":
    main()

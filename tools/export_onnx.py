#!/usr/bin/env python3
"""
Export an RSL-RL .pt checkpoint's actor network to ONNX.

Why: ONNX is the standard format for running a policy outside of PyTorch/Isaac
Lab entirely -- ONNX Runtime, TensorRT, a Jetson, a microcontroller toolchain,
etc. Like safetensors, it's a plain graph format (no pickle, no arbitrary code
execution on load), but unlike safetensors it's also a portable *computation
graph* -- something that can run with zero Python/PyTorch installed.

This exports only the deterministic policy (the action mean, no sampling) --
exactly what a trained policy actually runs at deployment time. The learned
action-noise parameter (std) used during training exploration is intentionally
left out of the graph; it's noted in metadata.json for completeness.

Usage:
    pip install onnx onnxruntime       # onnxruntime optional, only used to verify the export
    python export_onnx.py model_14990.pt --out ./vault_drop
        [--activation elu] [--opset 17] [--no-verify]

Outputs in --out (a directory, same convention as convert_checkpoint.py):
    actor.onnx          the exported graph
    metadata.json        merged into the existing sidecar if convert_checkpoint.py
                         already wrote one for this checkpoint, otherwise created fresh

Run this on the Tower, where the checkpoints live. Commit the outputs to
H1V3-RobotPolicies next to the original .pt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ACTIVATIONS = {
    "elu": "ELU",
    "relu": "ReLU",
    "lrelu": "LeakyReLU",
    "tanh": "Tanh",
    "sigmoid": "Sigmoid",
    "selu": "SELU",
    "softsign": "Softsign",
    "softplus": "Softplus",
}


def load_actor_state(ckpt: dict) -> dict:
    """Pull the actor's raw state dict out of either checkpoint save format (see
    convert_checkpoint.py's split_checkpoint for the full explanation of both formats)."""
    if "actor_state_dict" in ckpt:
        return ckpt["actor_state_dict"]
    if "model_state_dict" in ckpt:
        return {
            k[len("actor."):]: v
            for k, v in ckpt["model_state_dict"].items()
            if k.lower().startswith("actor.")
        }
    raise SystemExit(f"unrecognized checkpoint format (top-level keys: {list(ckpt.keys())})")


def build_mlp(obs_dim: int, action_dim: int, hidden_dims: list[int], activation: str):
    """Rebuild rsl_rl's MLP structure: Linear+activation repeated, final Linear with no
    activation. nn.Sequential auto-names children "0","1","2",... which matches the
    "mlp.0.weight"/"mlp.2.weight"/... keys in the checkpoint once the "mlp." prefix is
    stripped, so the state dict loads directly with no renaming."""
    import torch.nn as nn

    if activation not in ACTIVATIONS:
        raise SystemExit(f"unknown --activation '{activation}', expected one of {sorted(ACTIVATIONS)}")
    act_cls = getattr(nn, ACTIVATIONS[activation])

    dims = [obs_dim] + hidden_dims
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(act_cls())
    layers.append(nn.Linear(dims[-1], action_dim))
    return nn.Sequential(*layers)


def infer_dims(mlp_state: dict) -> tuple[int, int, list[int]]:
    """Derive obs_dim / action_dim / hidden_dims directly from the Linear layer shapes --
    no hardcoded per-robot assumption, works for any MLP size."""
    weight_items = sorted(
        ((k, v) for k, v in mlp_state.items() if k.endswith(".weight")),
        key=lambda kv: int(kv[0].split(".")[0]),
    )
    if not weight_items:
        raise SystemExit("no mlp.*.weight tensors found in the actor state dict")
    obs_dim = weight_items[0][1].shape[1]
    action_dim = weight_items[-1][1].shape[0]
    hidden_dims = [w.shape[0] for _, w in weight_items[:-1]]
    return obs_dim, action_dim, hidden_dims


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint", type=Path, help="input .pt checkpoint (your own file)")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--activation", default="elu", help="actor MLP activation (default: elu, this project's universal default)")
    ap.add_argument("--opset", type=int, default=18, help="ONNX opset version (default: 18 -- torch's current exporter only has native implementations for >=18 and silently downconverts otherwise)")
    ap.add_argument("--no-verify", action="store_true", help="skip the onnxruntime numerical cross-check")
    ap.add_argument("--robot", default="", help="robot name, only used if metadata.json doesn't already exist")
    args = ap.parse_args()

    import torch

    ckpt_path = args.checkpoint
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    # NOTE: weights_only=False -- same caveat as convert_checkpoint.py: only do this for your
    # own checkpoints, never an untrusted .pt.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise SystemExit(f"{ckpt_path} doesn't look like an RSL-RL checkpoint (type: {type(ckpt)})")

    actor_state = load_actor_state(ckpt)
    mlp_state = {k[len("mlp."):]: v for k, v in actor_state.items() if k.startswith("mlp.")}
    excluded_keys = sorted(k for k in actor_state.keys() if not k.startswith("mlp."))
    if not mlp_state:
        raise SystemExit("no mlp.* tensors found in the actor state dict")

    obs_dim, action_dim, hidden_dims = infer_dims(mlp_state)
    model = build_mlp(obs_dim, action_dim, hidden_dims, args.activation)
    model.load_state_dict(mlp_state, strict=True)
    model.eval()

    onnx_path = out / "actor.onnx"
    dummy_input = torch.zeros(1, obs_dim)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["observations"],
        output_names=["actions"],
        dynamic_axes={"observations": {0: "batch"}, "actions": {0: "batch"}},
        opset_version=args.opset,
    )

    # torch's exporter defaults to writing large tensors as a separate "external data" file
    # (actor.onnx.data next to actor.onnx) -- unnecessary for a model this small and makes the
    # .onnx file non-portable on its own. Reload with external data inlined and re-save as a
    # single self-contained file, then remove the now-orphaned external-data file.
    import onnx as onnx_lib

    external_data_path = Path(str(onnx_path) + ".data")
    onnx_model = onnx_lib.load(str(onnx_path), load_external_data=True)
    onnx_lib.save(onnx_model, str(onnx_path))
    if external_data_path.exists():
        external_data_path.unlink()

    props = {
        "obs_dim": str(obs_dim),
        "action_dim": str(action_dim),
        "hidden_dims": json.dumps(hidden_dims),
        "activation": args.activation,
        "source_file": ckpt_path.name,
        "output_is": "deterministic action mean (no exploration noise/sampling)",
    }
    for k, v in props.items():
        entry = onnx_model.metadata_props.add()
        entry.key, entry.value = k, v
    onnx_lib.save(onnx_model, str(onnx_path))

    verified = None
    max_abs_diff = None
    if not args.no_verify:
        try:
            import onnxruntime as ort
            import numpy as np

            torch.manual_seed(0)
            probe = torch.randn(8, obs_dim)
            with torch.no_grad():
                torch_out = model(probe).numpy()
            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            onnx_out = sess.run(None, {"observations": probe.numpy().astype("float32")})[0]
            max_abs_diff = float(np.abs(torch_out - onnx_out).max())
            verified = max_abs_diff < 1e-4
        except ImportError:
            print("NOTE: onnxruntime not installed, skipping numerical verification (pip install onnxruntime)")

    metadata_path = out / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {
        "robot": args.robot,
        "source_file": ckpt_path.name,
    }
    metadata["onnx_export"] = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_dims": hidden_dims,
        "activation": args.activation,
        "opset_version": args.opset,
        "excluded_actor_keys": excluded_keys,  # e.g. distribution.std_param -- training-only, not needed for deployment
        "verified_against_pytorch": verified,
        "max_abs_diff_vs_pytorch": max_abs_diff,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"wrote {onnx_path}")
    print(f"  obs_dim={obs_dim} action_dim={action_dim} hidden_dims={hidden_dims} activation={args.activation}")
    if verified is True:
        print(f"  verified against PyTorch forward pass: max abs diff = {max_abs_diff:.2e}")
    elif verified is False:
        print(f"  WARNING: mismatch vs PyTorch forward pass, max abs diff = {max_abs_diff:.2e}")
    else:
        print("  NOT verified (onnxruntime unavailable or --no-verify passed)")


if __name__ == "__main__":
    main()

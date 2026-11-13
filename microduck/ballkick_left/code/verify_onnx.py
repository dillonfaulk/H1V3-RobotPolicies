"""Check an exported actor ONNX against the PyTorch checkpoint it came from.

usage: python logs/verify_onnx.py <model_N.pt> <exported.onnx> [n_samples]
Rebuilds the actor (empirical normalizer (x - mean) / (std + 1e-2), ELU MLP) from
actor_state_dict and compares deterministic actions on random observations
(normalizer-scaled around the training mean). Prints the max abs difference.
"""
import sys

import numpy as np
import onnxruntime as ort
import torch

pt, onnx_path = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
sd = torch.load(pt, map_location="cpu", weights_only=False)["actor_state_dict"]
mean, std = sd["obs_normalizer._mean"], sd["obs_normalizer._std"]
layers = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.")})


def actor(x):
    h = (x - mean) / (std + 1e-2)
    for i, li in enumerate(layers):
        h = h @ sd[f"mlp.{li}.weight"].T + sd[f"mlp.{li}.bias"]
        if i < len(layers) - 1:
            h = torch.nn.functional.elu(h)
    return h


rng = np.random.default_rng(0)
obs = (mean.numpy() + rng.standard_normal((n, mean.shape[1])) * std.numpy() * 2).astype(np.float32)
sess = ort.InferenceSession(onnx_path)
name = sess.get_inputs()[0].name
print(f"onnx input {sess.get_inputs()[0].shape} -> output {sess.get_outputs()[0].shape}")
on = np.concatenate([sess.run(None, {name: o[None]})[0] for o in obs])
with torch.no_grad():
    tv = actor(torch.from_numpy(obs)).numpy()
print(f"max abs diff vs PyTorch over {n} obs: {np.abs(on - tv).max():.3g}")

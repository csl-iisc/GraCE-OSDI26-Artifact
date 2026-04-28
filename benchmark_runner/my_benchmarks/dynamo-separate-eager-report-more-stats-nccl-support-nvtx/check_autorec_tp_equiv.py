#!/usr/bin/env python3
"""
Equivalence check for NVIDIA DeepRecommender AutoEncoder:
  - TP=1 dense baseline vs TP=2 tensor-parallel wrapper
  - Works with or without torch.compile
  - Ensures identical weights & identical inputs on all ranks

Usage:
  # Dense reference (single GPU):
  CUDA_VISIBLE_DEVICES=0 python check_autorec_tp_equiv.py --tp_size 1

  # TP=2 (two GPUs):
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 check_autorec_tp_equiv.py --tp_size 2

Optional flags:
  --compile   : use torch.compile() for both models
  --dtype bf16|fp32|fp64  : math dtype (default fp32)
  --batch 8192            : batch size (default 8192 to match your traces)
  --layers 512,512,1024   : hidden layers (default matches your setup)
"""

import os
import argparse
import math
import torch
import torch.distributed as dist

# --- repo-local imports (paths you showed) ---
from torchbenchmark.models.nvidia_deeprecommender.reco_encoder.model.model import AutoEncoder
from torchbenchmark.models.nvidia_deeprecommender.tp_autorec import (
    apply_autorec_tensor_parallel,
    build_tp_group,
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tp_size", type=int, default=int(os.environ.get("TP_SIZE", "1")))
    p.add_argument("--compile", action="store_true")
    p.add_argument("--dtype", type=str, default="fp32", choices=["bf16", "fp32", "fp64"])
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--layers", type=str, default="512,512,1024")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()

def as_dtype(s: str):
    if s == "bf16": return torch.bfloat16
    if s == "fp64": return torch.float64
    return torch.float32

def tolerances_for_dtype(dt: torch.dtype):
    if dt == torch.bfloat16:
        return dict(rtol=1e-2, atol=3e-2)
    if dt == torch.float64:
        return dict(rtol=1e-8, atol=1e-10)
    return dict(rtol=1e-4, atol=3e-5)

def init_dist_if_needed(tp_size: int):
    if tp_size <= 1:
        return None, 0, 1
    if not dist.is_initialized():
        backend = "nccl"
        dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = dist.get_world_size()
    return dist.group.WORLD, local_rank, world_size

@torch.inference_mode()
def main():
    args = parse_args()
    dt = as_dtype(args.dtype)
    tol = tolerances_for_dtype(dt)

    # --- DDP/TP setup ---
    pg, local_rank, world_size = init_dist_if_needed(args.tp_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        # This model is massive; practical runs expect CUDA. We still support CPU for TP=1.
        if args.tp_size > 1:
            raise RuntimeError("CUDA is required for TP>1 in this checker.")
        device = torch.device("cpu")

    # --- Determinism: seed EVERYTHING the same on all ranks ---
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    import random, numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)

    # --- Model dims (match your setup) ---
    input_dim = 197951
    hidden_layers = [int(x) for x in args.layers.split(",")]  # e.g., [512, 512, 1024]
    layer_sizes = [input_dim] + hidden_layers
    batch = args.batch

    # --- Construct a base weight set deterministically ---
    base = AutoEncoder(
        layer_sizes=layer_sizes,
        nl_type="selu",
        is_constrained=False,      # TP wrapper expects non-constrained
        dp_drop_prob=0.0,          # avoid randomness
        last_layer_activations=True
    ).to(device=device, dtype=dt).eval()

    # Clone baseline with identical weights
    dense_ref = AutoEncoder(
        layer_sizes=layer_sizes,
        nl_type="selu",
        is_constrained=False,
        dp_drop_prob=0.0,
        last_layer_activations=True
    ).to(device=device, dtype=dt).eval()
    dense_ref.load_state_dict(base.state_dict(), strict=True)

    # --- Build TP wrapper from the same weights (only if tp_size>1) ---
    if args.tp_size > 1:
        # create TP subgroup (WORLD is fine for pure-TP)
        tp_group = build_tp_group(args.tp_size)
        tp_model = apply_autorec_tensor_parallel(base, tp_group, verbose=(local_rank == 0))
        tp_model = tp_model.to(device=device, dtype=dt).eval()
    else:
        tp_model = None  # not used in dense-only run

    # --- Compile (optional) ---
    if args.compile:
        dense_ref = torch.compile(dense_ref, fullgraph=False, dynamic=False)
        if tp_model is not None:
            tp_model = torch.compile(tp_model, fullgraph=False, dynamic=False)

    # --- Create one deterministic input and broadcast to all ranks ---
    g = torch.Generator(device=device).manual_seed(args.seed + 99)
    x = torch.randn(batch, input_dim, generator=g, device=device, dtype=dt)
    if args.tp_size > 1:
        dist.broadcast(x, src=0)

    # --- Run dense reference ON RANK 0 (then broadcast) ---
    if args.tp_size > 1 and dist.get_rank() != 0:
        ref_out = torch.empty(batch, input_dim, device=device, dtype=dt)
    else:
        ref_out = dense_ref(x)
    if args.tp_size > 1:
        dist.broadcast(ref_out, src=0)

    # --- Run TP model or dense again (TP=1 path) ---
    if tp_model is None:
        test_out = dense_ref(x)  # sanity: compare dense vs dense (should be identical)
    else:
        test_out = tp_model(x)

    # --- Compare on every rank ---
    same = torch.allclose(test_out, ref_out, **tol)
    max_abs = (test_out - ref_out).abs().max().item()
    mean_abs = (test_out - ref_out).abs().mean().item()

    msg = (
        f"[rank {local_rank}/{world_size}] "
        f"allclose={same} rtol={tol['rtol']} atol={tol['atol']} "
        f"max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} "
        f"(dtype={dt}, batch={batch}, layers={hidden_layers})"
    )
    print(msg)

    # If TP>1, also sanity-check that all ranks produced identical TP outputs
    if args.tp_size > 1:
        # gather a single scalar checksum
        cs_local = float(test_out.float().sum().item())
        cs_tensor = torch.tensor([cs_local], device=device)
        cs_all = [torch.empty_like(cs_tensor) for _ in range(world_size)]
        dist.all_gather(cs_all, cs_tensor)
        eq_among_ranks = all(abs(cs_all[0].item() - t.item()) < 1e-6 for t in cs_all[1:])
        if local_rank == 0:
            print(f"[tp consistency] outputs equal across ranks: {eq_among_ranks}")

    # Exit code (handy in CI)
    if args.tp_size > 1 and dist.get_rank() == 0:
        # Print a crisp pass/fail summary on rank0
        print("[SUMMARY] ", "PASS" if same else "FAIL")

    # Clean up
    if args.tp_size > 1 and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()

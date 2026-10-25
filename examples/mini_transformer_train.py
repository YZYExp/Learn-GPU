#!/usr/bin/env python3
"""A complete tiny causal Decoder training lab; synthetic data, no downloads.

Requires a compatible recent PyTorch installation (SDPA and optional compile).
Default: CPU FP32. CUDA BF16 is optional; FP16 is intentionally unsupported.
Example: python3 examples/mini_transformer_train.py --steps 20
Validated on CPU with PyTorch 2.14.1+cpu: training, compile, accumulation,
activation recomputation, and exact split-run checkpoint restoration.
CUDA/BF16 paths have not been executed by the book authors.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time
from contextlib import nullcontext


DEFAULTS = {
    "batch": 8, "seq": 32, "dim": 64, "heads": 4, "layers": 2,
    "vocab": 32, "lr": 0.002, "seed": 20261025, "precision": "fp32",
    "compile": False, "accum": 1, "activation_checkpoint": False,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20,
                        help="Total completed optimizer steps, including resumed steps.")
    for name in ("batch", "seq", "dim", "heads", "layers", "vocab", "seed", "accum"):
        parser.add_argument(f"--{name}", type=int, default=None,
                            help=f"Default {DEFAULTS[name]}; inherited on resume.")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None,
                        help="cpu (default), cuda, cuda:0, or auto; inherited on resume.")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default=None)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--activation-checkpoint", action=argparse.BooleanOptionalAction,
                        default=None, help="Recompute each Decoder block during backward.")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Output checkpoint; default resume path or build/mini-transformer.pt.")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--save-every", type=int, default=0,
                        help="Save every N complete updates; 0 saves only at normal completion.")
    return parser.parse_args()


def resolve_config(args, payload):
    saved = payload["config"] if payload else None
    cfg = {}
    for key, default in DEFAULTS.items():
        requested = getattr(args, key)
        if saved is not None:
            if key not in saved:
                raise ValueError(f"Checkpoint lacks config key {key!r}")
            if requested is not None and requested != saved[key]:
                raise ValueError(f"Resume config mismatch for {key}: "
                                 f"saved={saved[key]!r}, requested={requested!r}")
            cfg[key] = saved[key]
        else:
            cfg[key] = default if requested is None else requested
    for key in ("batch", "seq", "dim", "heads", "layers", "vocab", "accum"):
        if cfg[key] <= 0:
            raise ValueError(f"--{key} must be positive")
    if cfg["vocab"] < 2 or cfg["dim"] % cfg["heads"] or cfg["lr"] <= 0:
        raise ValueError("Need vocab >= 2, dim divisible by heads, and lr > 0")
    if args.steps <= 0 or args.save_every < 0:
        raise ValueError("Need --steps > 0 and --save-every >= 0")
    return cfg


def make_model(torch, cfg):
    nn = torch.nn
    functional = torch.nn.functional
    from torch.utils.checkpoint import checkpoint as activation_checkpoint

    class CausalAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = cfg["heads"]
            self.head_dim = cfg["dim"] // self.heads
            self.qkv = nn.Linear(cfg["dim"], 3 * cfg["dim"])
            self.output = nn.Linear(cfg["dim"], cfg["dim"])

        def forward(self, x):
            batch, seq, dim = x.shape
            qkv = self.qkv(x).reshape(batch, seq, 3, self.heads, self.head_dim)
            # Each of q/k/v is [batch, heads, sequence, head_dim].
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            out = functional.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=True)
            out = out.transpose(1, 2).contiguous().reshape(batch, seq, dim)
            return self.output(out)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(cfg["dim"])
            self.attention = CausalAttention()
            self.norm2 = nn.LayerNorm(cfg["dim"])
            self.mlp = nn.Sequential(nn.Linear(cfg["dim"], 4 * cfg["dim"]),
                                     nn.GELU(),
                                     nn.Linear(4 * cfg["dim"], cfg["dim"]))

        def forward(self, x):
            x = x + self.attention(self.norm1(x))
            return x + self.mlp(self.norm2(x))

    class TinyDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.tokens = nn.Embedding(cfg["vocab"], cfg["dim"])
            self.positions = nn.Embedding(cfg["seq"], cfg["dim"])
            self.blocks = nn.ModuleList([Block() for _ in range(cfg["layers"])])
            self.final_norm = nn.LayerNorm(cfg["dim"])
            self.head = nn.Linear(cfg["dim"], cfg["vocab"], bias=False)

        def forward(self, token_ids):
            positions = torch.arange(token_ids.shape[1], device=token_ids.device)
            x = self.tokens(token_ids) + self.positions(positions)
            for block in self.blocks:
                if cfg["activation_checkpoint"] and self.training:
                    x = activation_checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)
            return self.head(self.final_norm(x))

    return TinyDecoder()


def next_batch(torch, cfg, generator, device):
    # CPU generator is independent of global model/compiled RNG state.
    starts = torch.randint(cfg["vocab"], (cfg["batch"], 1), generator=generator)
    positions = torch.arange(cfg["seq"] + 1).unsqueeze(0)
    sequence = (starts + positions) % cfg["vocab"]
    inputs = sequence[:, :-1].contiguous().to(device)
    targets = sequence[:, 1:].contiguous().to(device)
    return inputs, targets


def synchronize(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def save_checkpoint(torch, path, model, optimizer, generator, step, cfg, device):
    # Called only after optimizer.step() and device synchronization.
    payload = {
        "format": 1, "optimizer_step_complete": True,
        "step": step, "config": dict(cfg), "device": str(device),
        "torch_version": str(torch.__version__),
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "data_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        "python_rng": random.getstate(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    args = parse_args()  # --help works even before PyTorch is installed.
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Install a compatible PyTorch build before running this lab.") from exc

    payload = None
    if args.resume is not None:
        payload = torch.load(args.resume, map_location="cpu", weights_only=True)
        if payload.get("format") != 1 or not payload.get("optimizer_step_complete"):
            raise ValueError("Not a complete optimizer-step checkpoint from this lab")
    cfg = resolve_config(args, payload)
    requested_device = args.device or (payload["device"] if payload else "cpu")
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("This lab supports CPU and CUDA only")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA is unavailable in this PyTorch environment")
    if device.type == "cpu" and device.index is not None:
        raise ValueError("Use --device cpu without an index")
    if cfg["precision"] == "bf16":
        if device.type != "cuda":
            raise ValueError("This lab limits BF16 autocast to supported CUDA devices")
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("Requested CUDA device does not support BF16")
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg["seed"])
        # FP32 label means ordinary FP32 matmul here, without optional TF32.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    generator = torch.Generator(device="cpu").manual_seed(cfg["seed"] + 1)
    model = make_model(torch, cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    completed = 0
    if payload is not None:
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        generator.set_state(payload["data_rng"])
        completed = int(payload["step"])
        if args.steps < completed:
            raise ValueError("--steps is a total target and cannot precede the saved step")
        if str(torch.__version__) != payload["torch_version"]:
            print("NOTICE: PyTorch version changed; numerical continuity is not guaranteed.")
        if str(device) != payload["device"]:
            print("NOTICE: Device changed; identical data order does not imply identical arithmetic.")
    model.train()

    def forward_loss(inputs, targets):
        # Put autocast INSIDE the compiled region. Its ambient entry/exit context
        # is no-autocast, matching backward() below, including AOTAutograd's
        # documented same-as-forward default assumption.
        context = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                   if cfg["precision"] == "bf16" else nullcontext())
        with context:
            logits = model(inputs)
            return torch.nn.functional.cross_entropy(
                logits.reshape(-1, cfg["vocab"]), targets.reshape(-1))

    objective = torch.compile(forward_loss) if cfg["compile"] else forward_loss
    # Restore after construction/loading so initialization cannot advance the
    # resumed random stream. Compilation caches are intentionally not serialized.
    if payload is not None:
        random.setstate(payload["python_rng"])
        torch.set_rng_state(payload["torch_rng"])
        if device.type == "cuda" and payload["cuda_rng"] is not None:
            torch.cuda.set_rng_state(payload["cuda_rng"], device)
    checkpoint_path = args.checkpoint or args.resume or Path("build/mini-transformer.pt")
    print(json.dumps({"config": cfg, "device": str(device),
                      "torch": str(torch.__version__), "resume_step": completed,
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "timing": "data generation + transfer + accumulated fwd/bwd + update; "
                                "synchronized; excludes logging and checkpoint I/O"},
                     ensure_ascii=False))
    first_this_process = True
    for step in range(completed + 1, args.steps + 1):
        synchronize(torch, device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        step_loss = torch.zeros((), device=device)
        for _ in range(cfg["accum"]):
            inputs, targets = next_batch(torch, cfg, generator, device)
            loss = objective(inputs, targets)
            step_loss = step_loss + loss.detach() / cfg["accum"]
            (loss / cfg["accum"]).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0,
                                       error_if_nonfinite=True)
        optimizer.step()
        synchronize(torch, device)
        elapsed = time.perf_counter() - start
        record = {"step": step, "loss": float(step_loss), "step_ms": elapsed * 1000,
                  "tokens_per_step": cfg["batch"] * cfg["seq"] * cfg["accum"],
                  "tokens_per_second": cfg["batch"] * cfg["seq"] * cfg["accum"] / elapsed,
                  "first_update_in_process": first_this_process,
                  "may_include_compilation": bool(first_this_process and cfg["compile"])}
        if device.type == "cuda":
            record["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
            record["peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2**20
        print(json.dumps(record, ensure_ascii=False))
        first_this_process = False
        completed = step
        if args.save_every and step % args.save_every == 0:
            save_checkpoint(torch, checkpoint_path, model, optimizer, generator,
                            completed, cfg, device)
    # No mid-accumulation snapshots; a crash retains only the last atomic save.
    synchronize(torch, device)
    save_checkpoint(torch, checkpoint_path, model, optimizer, generator,
                    completed, cfg, device)
    print(f"Saved completed optimizer step {completed} to {checkpoint_path}")


if __name__ == "__main__":
    main()

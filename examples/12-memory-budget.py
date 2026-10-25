"""Analytical training-state budget, not a measurement of a framework."""
import argparse
import json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--params", type=int, default=1_000_000_000)
parser.add_argument("--param-bytes", type=float, default=2)
parser.add_argument("--grad-bytes", type=float, default=2)
parser.add_argument("--optimizer-bytes", type=float, default=8)
parser.add_argument("--master-bytes", type=float, default=4)
args = parser.parse_args()
if args.params < 0 or min(args.param_bytes, args.grad_bytes,
                          args.optimizer_bytes, args.master_bytes) < 0:
    parser.error("parameter count and byte costs must be nonnegative")
parts = {"parameters": args.param_bytes, "gradients": args.grad_bytes,
         "optimizer": args.optimizer_bytes, "master_copy": args.master_bytes}
result = {name: args.params * cost for name, cost in parts.items()}
result["total_bytes"] = sum(result.values())
result["total_GB_decimal"] = result["total_bytes"] / 1e9
result["total_GiB_binary"] = result["total_bytes"] / 2**30
result["excluded"] = ["activations", "workspaces", "communication buffers",
                      "allocator overhead", "framework and driver allocations"]
print(json.dumps(result, indent=2))

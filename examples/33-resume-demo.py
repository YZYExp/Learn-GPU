#!/usr/bin/env python3
"""CPU-only checkpoint lesson: weights + momentum + step + RNG, no dependencies.

Run: python3 examples/33-resume-demo.py
A complete JSON checkpoint must reproduce this toy program's uninterrupted run.
This demonstrates state semantics; it is not distributed checkpoint software.
"""
import copy
import json
from pathlib import Path
import random
import tempfile


def initial_state():
    return {"step": 0, "w": 0.0, "b": 0.0, "mw": 0.0, "mb": 0.0}


def advance(state, rng, target_step):
    for _ in range(state["step"], target_step):
        # Sampling order is part of the training state.
        xs = [rng.uniform(-1, 1) for _ in range(8)]
        errors = [state["w"] * x + state["b"] - (3 * x + 1) for x in xs]
        gw = sum(2 * e * x for e, x in zip(errors, xs)) / len(xs)
        gb = sum(2 * e for e in errors) / len(xs)
        state["mw"] = 0.9 * state["mw"] + gw
        state["mb"] = 0.9 * state["mb"] + gb
        state["w"] -= 0.03 * state["mw"]
        state["b"] -= 0.03 * state["mb"]
        state["step"] += 1
    return state


def tuples(value):
    # JSON preserves values but represents Python tuple state as lists.
    return tuple(tuples(x) for x in value) if isinstance(value, list) else value


def main():
    baseline = advance(initial_state(), random.Random(73), 40)
    rng = random.Random(73)
    paused = advance(initial_state(), rng, 15)
    snapshot = {"format_version": 1, "training": paused, "rng": rng.getstate()}
    with tempfile.TemporaryDirectory() as temp:
        final = Path(temp) / "checkpoint.json"
        staging = Path(temp) / "checkpoint.partial"
        staging.write_text(json.dumps(snapshot), encoding="utf-8")
        # Atomic rename on the same local filesystem; not a promise for object stores.
        staging.replace(final)
        loaded = json.loads(final.read_text(encoding="utf-8"))
    restored_rng = random.Random()
    restored_rng.setstate(tuples(loaded["rng"]))
    resumed = advance(copy.deepcopy(loaded["training"]), restored_rng, 40)
    assert resumed == baseline, "Complete toy checkpoint did not reproduce baseline"
    # Deliberately remove optimizer momentum while preserving sample order.
    incomplete = copy.deepcopy(loaded["training"])
    incomplete["mw"] = incomplete["mb"] = 0.0
    restored_rng.setstate(tuples(loaded["rng"]))
    wrong = advance(incomplete, restored_rng, 40)
    assert wrong != baseline, "The missing-state demonstration unexpectedly matched"
    print("完整恢复：与本进程中连续训练的状态完全一致")
    print(f"连续/完整恢复：w={baseline['w']:.8f}, b={baseline['b']:.8f}")
    print(f"丢失动量后恢复：w={wrong['w']:.8f}, b={wrong['b']:.8f}")
    print("此结果只验证本CPU示例；不代表跨设备或分布式逐位复现。")


if __name__ == "__main__":
    main()

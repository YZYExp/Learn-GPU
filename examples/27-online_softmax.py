#!/usr/bin/env python3
"""CPU demonstration of tiled online softmax weighted sums.

Run: python3 examples/27-online_softmax.py
Standard library only. This validates the forward mathematical invariant;
it is not a GPU kernel, benchmark, gradient implementation, or dropout model.
Scores must be finite. All-masked rows need separate interface-specific handling.
"""
import math
import random


def reference(scores: list[float], values: list[list[float]]) -> list[float]:
    maximum = max(scores)
    exp_scores = [math.exp(x - maximum) for x in scores]
    denominator = math.fsum(exp_scores)
    return [math.fsum(w * row[j] for w, row in zip(exp_scores, values))
            / denominator for j in range(len(values[0]))]


def online(scores: list[float], values: list[list[float]], tile: int) -> list[float]:
    if tile <= 0 or not scores or len(scores) != len(values):
        raise ValueError("Need a positive tile and non-empty matching input rows")
    width = len(values[0])
    if width == 0 or any(len(v) != width for v in values):
        raise ValueError("Values must have a fixed positive width")
    if not all(math.isfinite(x) for x in scores):
        raise ValueError("This demonstration accepts finite scores only")
    maximum = -math.inf
    denominator = 0.0
    numerator = [0.0] * width
    for start in range(0, len(scores), tile):
        block_scores = scores[start:start + tile]
        block_values = values[start:start + tile]
        new_maximum = max(maximum, max(block_scores))
        old_scale = math.exp(maximum - new_maximum)
        weights = [math.exp(x - new_maximum) for x in block_scores]
        denominator = old_scale * denominator + math.fsum(weights)
        numerator = [old_scale * numerator[j]
                     + math.fsum(w * row[j]
                                 for w, row in zip(weights, block_values))
                     for j in range(width)]
        maximum = new_maximum
    return [x / denominator for x in numerator]


def main() -> None:
    # A two-block worked example has an independently known answer 50/3.
    answer = online([0, math.log(2)], [[10.0], [20.0]], 1)[0]
    assert math.isclose(answer, 50 / 3, rel_tol=1e-14)
    print(f"Two-block answer: {answer:.12f}; expected 50/3")
    rng = random.Random(20261025)
    worst_error = 0.0
    for n in [1, 2, 7, 31, 129]:
        scores = [rng.uniform(-1000, 1000) for _ in range(n)]
        values = [[rng.uniform(-3, 3) for _ in range(5)] for _ in range(n)]
        expected = reference(scores, values)
        for tile in [1, 3, 16, 64, 256]:
            actual = online(scores, values, tile)
            error = max(abs(a - b) for a, b in zip(actual, expected))
            worst_error = max(worst_error, error)
            assert all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                       for a, b in zip(actual, expected))
    print(f"CPU forward checks passed; maximum absolute error={worst_error:.3g}")
    print("No GPU timing or backward validation was performed.")


if __name__ == "__main__":
    main()

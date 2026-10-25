#!/usr/bin/env python3
"""Theoretical estimates only. No GPU dependency; not a benchmark.

Run: python3 examples/25-roofline_estimator.py
Decimal SI units are used: TB/s=1e12 bytes/s, TFLOP/s=1e12 FLOP/s.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Roofline:
    bandwidth_tb_s: float
    compute_tflop_s: float

    def ceiling_tflop_s(self, intensity_flop_byte: float) -> float:
        if min(self.bandwidth_tb_s, self.compute_tflop_s, intensity_flop_byte) <= 0:
            raise ValueError("All roofline inputs must be positive")
        return min(self.compute_tflop_s,
                   self.bandwidth_tb_s * intensity_flop_byte)


def amdahl_speedup(fraction: float, local_speedup: float) -> float:
    if not 0 <= fraction <= 1 or local_speedup <= 0:
        raise ValueError("fraction must be in [0, 1]; speedup must be positive")
    return 1 / (1 - fraction + fraction / local_speedup)


def main() -> None:
    # Hypothetical device and matched FP32 compute ceiling, NOT product specs.
    roof = Roofline(bandwidth_tb_s=1.0, compute_tflop_s=20.0)
    intensity = 1 / 12  # Two FP32 reads + one FP32 write, one addition.
    bound = roof.ceiling_tflop_s(intensity)
    print("Theoretical FP32 vector-add arithmetic intensity:", intensity)
    print(f"Theoretical roofline ceiling: {bound * 1000:.3f} GFLOP/s")
    print("No hardware measurement was performed.")
    print("50% of an iteration accelerated 2x:", amdahl_speedup(0.5, 2))
    # Independent closed-form checks for the worked examples in chapter 25.
    assert abs(bound * 1000 - 1000 / 12) < 1e-10
    assert abs(amdahl_speedup(0.5, 2) - 4 / 3) < 1e-12
    print("Worked-example checks passed.")


if __name__ == "__main__":
    main()

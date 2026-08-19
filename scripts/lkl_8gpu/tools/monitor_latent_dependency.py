#!/usr/bin/env python3
"""Track latent-dependency indicators across a stochastic-SFT training log.

Reads the training log and prints a compact trend of the per-100-latent-step
metrics: noise_std_mean (exploration strength) and answer_grad_norm (answer
loss gradient w.r.t. the final latent, i.e. how much the answer depends on it).

Usage:
  python scripts/lkl_8gpu/tools/monitor_latent_dependency.py <train.log>
"""

from __future__ import annotations

import re
import sys


PAT = re.compile(
    r"\[latent-stochastic\] step=(\d+) noise_std_mean=([0-9.]+) "
    r"alpha=([0-9.]+) answer_grad_norm=([0-9.eE+-]+|n/a)"
)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    log_path = sys.argv[1]

    rows = []
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = PAT.search(line)
            if m:
                rows.append(
                    (
                        int(m.group(1)),
                        float(m.group(2)),
                        float(m.group(3)),
                        None if m.group(4) == "n/a" else float(m.group(4)),
                    )
                )

    if not rows:
        print("No [latent-stochastic] entries found in", log_path)
        return

    print(f"entries={len(rows)}  latent_steps={rows[-1][0]}")
    print(f"{'latent_step':>12s} {'noise_std':>10s} {'grad_norm':>12s}")
    # Aggregate per latent-step bucket (each bucket has 8 rank entries).
    buckets: dict[int, list[float]] = {}
    for step, noise, _alpha, grad in rows:
        buckets.setdefault(step, []).append((noise, grad))
    for step in sorted(buckets):
        entries = buckets[step]
        noise_avg = sum(e[0] for e in entries) / len(entries)
        grads = [e[1] for e in entries if e[1] is not None]
        grad_avg = sum(grads) / len(grads) if grads else float("nan")
        grad_max = max(grads) if grads else float("nan")
        print(f"{step:>12d} {noise_avg:>10.4f} {grad_avg:>12.6f} (max {grad_max:.6f})")

    print("\nInterpretation: grad_norm << 1e-3 throughout suggests the answer is not")
    print("depending on the latent trajectory (bypass); a rising trend indicates")
    print("dependence is being established. Cross-check with intervention eval.")


if __name__ == "__main__":
    main()

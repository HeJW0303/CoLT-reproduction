#!/usr/bin/env python3
"""Merge per-shard reward-gate outputs into the global decision.

Each shard writes per-sample metrics (answer grad norms, per-sample answer-CE
std, grounding-loss std, and per-sample grounding<->answer coupling). This
script concatenates them and recomputes the global aggregates and decision.

Usage:
  python scripts/lkl_8gpu/tools/merge_gate_shards.py --inputs shard_0.json shard_1.json ... --out final.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    shards = [json.load(open(p, encoding="utf-8")) for p in args.inputs]
    if not shards:
        raise SystemExit("no shard files given")

    # A merged decision is only meaningful when all measurements used the same
    # checkpoint and gate semantics.  Keep legacy artifacts mergeable (their
    # protocol fields are uniformly absent), but reject accidental mixtures.
    for field in (
        "checkpoint",
        "k",
        "sigmas",
        "protocol",
        "gradient_capture",
        "loss_isolation",
    ):
        field_values = {json.dumps(shard.get(field), sort_keys=True) for shard in shards}
        if len(field_values) != 1:
            raise SystemExit(f"incompatible shard field {field!r}: {sorted(field_values)}")

    sigmas = shards[0]["sigmas"]
    grads = [g for s in shards for g in s.get("per_sample_grads", [])]
    ce_std = {f"{sig:.2f}": [v for s in shards for v in s["per_sample_answer_ce_std"][f"{sig:.2f}"]]
              for sig in sigmas}
    gr_std = {f"{sig:.2f}": [v for s in shards for v in s["per_sample_grounding_loss_std"][f"{sig:.2f}"]]
              for sig in sigmas}
    noise_reach = {
        f"{sig:.2f}": [v for s in shards for v in s.get("per_sample_noise_reach", {}).get(f"{sig:.2f}", [])]
        for sig in sigmas
    }
    couplings = {
        f"{sig:.2f}": [
            value
            for shard in shards
            for value in shard.get("per_sample_couplings_by_sigma", {}).get(f"{sig:.2f}", [])
        ]
        for sig in sigmas
    }
    # Historical v1 shards kept one mixed list across all sigmas.  Retain a
    # readable merge path for old artifacts, but new v2 gates always populate
    # the per-sigma structure above.
    if not any(couplings.values()):
        positive_sigmas = [sigma for sigma in sigmas if sigma > 0.0]
        fallback_key = f"{positive_sigmas[0] if positive_sigmas else sigmas[0]:.2f}"
        couplings[fallback_key] = [
            value for shard in shards for value in shard.get("per_sample_couplings", [])
        ]

    def med(vals: list[float]) -> float:
        return statistics.median(vals) if vals else float("nan")

    grad_median = med(grads)
    grad_mean = statistics.mean(grads) if grads else float("nan")
    ce_std_med = {k: med(v) for k, v in ce_std.items()}
    gr_std_med = {k: med(v) for k, v in gr_std.items()}
    noise_reach_med = {k: med(v) for k, v in noise_reach.items()}
    coupling_mean_by_sigma = {
        key: statistics.mean(values) if values else float("nan")
        for key, values in couplings.items()
    }
    coupling_pos_by_sigma = {
        key: sum(1 for value in values if value > 0) / max(len(values), 1)
        for key, values in couplings.items()
    }
    grad_used_frac = sum(1 for g in grads if g > 0) / max(len(grads), 1)

    sigma_ref = next(sigma for sigma in sigmas if sigma > 0.0)
    sigma_ref_key = f"{sigma_ref:.2f}"
    expected_delta = grad_median * sigma_ref * math.sqrt(2 / math.pi)
    answer_graph_connected = grad_used_frac > 0.0
    answer_response_present = ce_std_med[sigma_ref_key] > 1e-8 and noise_reach_med[sigma_ref_key] > 1e-4
    grounding_response_present = gr_std_med[sigma_ref_key] > 1e-8
    coupling_mean = coupling_mean_by_sigma[sigma_ref_key]
    coupling_pos = coupling_pos_by_sigma[sigma_ref_key]
    coupling_ok = (coupling_mean > 0.1) or (coupling_pos > 0.55)
    answer_alive = answer_graph_connected and answer_response_present
    grounding_alive = grounding_response_present and coupling_ok

    if not answer_graph_connected:
        decision = "SFT: answer CE is graph-disconnected from the final latent"
    elif not answer_response_present:
        decision = "SFT: final latent is graph-connected but answer CE is insensitive to validated path noise"
    elif not grounding_alive:
        decision = "SFT: answer path responds, but grounding reward lacks stable answer-coupled discrimination"
    else:
        decision = "RL candidate: answer and grounding rewards both respond under the gate"

    result = {
        "protocol": shards[0].get("protocol"),
        "gradient_capture": shards[0].get("gradient_capture"),
        "loss_isolation": shards[0].get("loss_isolation"),
        "checkpoint": shards[0].get("checkpoint"),
        "n_shards": len(shards),
        "n_answer_total": len(grads),
        "n_ground_total": len(couplings[sigma_ref_key]),
        "k": shards[0].get("k"),
        "sigmas": sigmas,
        "answer_grad_norm_median": grad_median,
        "answer_grad_norm_mean": grad_mean,
        "answer_grad_used_frac": grad_used_frac,
        "answer_graph_connected": answer_graph_connected,
        "answer_response_present": answer_response_present,
        "grounding_response_present": grounding_response_present,
        "answer_response_sigma": sigma_ref,
        "expected_answer_delta_at_response_sigma": expected_delta,
        "answer_ce_std_median_by_sigma": ce_std_med,
        "grounding_loss_std_median_by_sigma": gr_std_med,
        "noise_reach_median_by_sigma": noise_reach_med,
        "coupling_pearson_mean": coupling_mean,
        "coupling_positive_frac": coupling_pos,
        "coupling_pearson_mean_by_sigma": coupling_mean_by_sigma,
        "coupling_positive_frac_by_sigma": coupling_pos_by_sigma,
        # Preserve the raw measurements needed to audit a threshold decision;
        # summary statistics alone cannot show whether an average is driven by
        # a small set of outliers.
        "per_sample_grads": grads,
        "per_sample_answer_ce_std": ce_std,
        "per_sample_grounding_loss_std": gr_std,
        "per_sample_noise_reach": noise_reach,
        "per_sample_couplings": couplings[sigma_ref_key],
        "per_sample_couplings_by_sigma": couplings,
        "answer_alive": answer_alive,
        "grounding_alive": grounding_alive,
        "coupling_ok": coupling_ok,
        "decision": decision,
    }
    print(json.dumps(result, indent=1, ensure_ascii=False))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()

# Vendored EasyR1 provenance

This directory is vendored from the `EasyR1/` subtree of OneThinker.

- Upstream repository: `https://github.com/tulerfeng/OneThinker.git`
- Upstream commit: `4a36ad286d04382fce9816ac4429e650157a5f11`
- Imported subtree: `EasyR1/`
- Import date: 2026-08-08
- License: Apache-2.0; see `LICENSE`

The subtree is intentionally vendored because CoLT requires coordinated changes to the
rollout backend, FSDP weight synchronization, actor log-probability computation, prompt
format, and reward function. Keep upstream synchronization separate from CoLT-specific
changes so that provenance and behavioral changes remain reviewable.

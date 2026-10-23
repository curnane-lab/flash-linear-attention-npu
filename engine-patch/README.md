# Engine-side patch: GDN recover verify wiring (vllm-ascend)

Source: `/data1/f00538480/vllm-ascend-sketchssm`, branch `feat/gdn-recover-verify`,
commits `83373d8..76f9f0e` (6 commits, format-patch series in this directory).

Content: experimental GDN recover wiring for the spec-decode verify path —
`VLLM_ASCEND_GDN_RECOVER=1` routes pure spec-decode verify through
`aclnnRecurrentGatedDeltaRuleRecover` with a per-layer records workspace;
startup capability probe (refuses wrong OPP ordering); env-gated per-call
records dump (`VLLM_ASCEND_GDN_RECOVER_DUMP`); phase-bucketed call ledger.

Apply: `git am engine-patch/000*.patch` on top of the source tree at `83373d8`.

Companion evidence (experiment tree, not in this repo):
`evaluation/ascend910/experiments/gdn_recover_public_primitives_20261021/r0_20261021/`
`n10/` (real-engine A/B), `n11/` (records oracle + OPP guard), `n12/` (copy
diagnosis), `n13/` (R0 contract / R1 records-only variant / R2 cost /
allocator audit).

Note (2026-10-23): the records-only performance line was measured
not-advantageous and archived; the capacity mechanism belongs to upstream
vllm-ascend #18765. This patch series is preserved for reference/regrouping.

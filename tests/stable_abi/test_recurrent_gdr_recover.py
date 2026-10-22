#!/usr/bin/env python3
"""Regression tests for the RecurrentGatedDeltaRule recover extension.

Covers the public Stable-ABI entry ``npu_recurrent_gated_delta_rule_recover``
(and the two-stage aclnn entry via ``fla_npu.ops.ascendc._runtime``) against an
exact, independent fp64 CPU reference:

1. numeric spot cases: L4/B4/H16/HV32 fp32 state, a single-token request, and
   the accepted-token read-row offset variant;
2. read-only ``source_state`` bit preservation and the sentinel on pool slots /
   records rows/steps the call must not write;
3. host-side rejections: non-contiguous ``source_state`` inner dims, undersized
   records step capacity (< MAX_MTP=8), ``source_state``/``state`` storage
   aliasing, ``gk``+``records`` pairing, ``record_indices`` without ``records``;
4. mutation contract: one call bumps the ``state`` and ``records`` version
   counters exactly once, and grad-requiring tensors are refused.

Primitive-graph replay is NOT_APPLICABLE here: this harness has no
graph-capture facility for this op family; replay bit-equality of the shipped
op was measured separately with an experiment-side probe.

Run after installing the current build (see tests/stable_abi/README.md):
    python tests/stable_abi/test_recurrent_gdr_recover.py --device 0
"""
import argparse
import ctypes

import torch
import torch_npu  # noqa: F401

from fla_npu.ops.ascendc import npu_recurrent_gated_delta_rule_recover

K = V = 128
MAX_MTP = 8
SENTINEL = -777.0

TOL_OUT = 5e-3    # bf16 output vs fp64 oracle
TOL_STATE = 1e-3  # fp32 state vs fp64 oracle
TOL_DC = 1e-3     # fp32 dc record
TOL_ALPHA = 1e-5  # fp32 alpha = exp(g)

FAILURES = []


def report(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f" ({detail})" if detail else ""),
          flush=True)
    if not ok:
        FAILURES.append(name)


def rel(a, b):
    return float((a.double() - b.double()).norm()
                 / b.double().norm().clamp_min(1e-12))


def reference_recover(q, k, v, g, beta, source, indices, record_indices,
                      lengths, scale, dest_slots, accepted=None):
    """fp64 reference: delta-rule recurrence with recover bookkeeping.

    The read row of request b is indices[start_b + accepted - 1] when
    ``accepted`` is set, else indices[start_b]; ``records[r, head, step]``
    packs [V] dc | [K] key | 1 alpha per token.
    """
    T = q.shape[0]
    H, HV = q.shape[1], v.shape[1]
    kidx = [hv // (HV // H) for hv in range(HV)]
    out = torch.zeros(T, HV, V, dtype=torch.float64)
    dest = torch.full((dest_slots, HV, V, K), float("nan"), dtype=torch.float64)
    Lmax = max(lengths)
    records = torch.full((int(max(record_indices)) + 1, HV, Lmax, V + K + 1),
                         float("nan"), dtype=torch.float64)
    off = 0
    for b, L in enumerate(lengths):
        read_i = off if accepted is None else off + accepted - 1
        M = source[int(indices[read_i])].clone()
        for i in range(L):
            t = off + i
            a = g[t].exp().view(HV, 1)
            kk = k[t][kidx]
            dc = beta[t].view(HV, 1) * (v[t] - a * torch.einsum("hvk,hk->hv", M, kk))
            M = a.view(HV, 1, 1) * M + torch.einsum("hv,hk->hvk", dc, kk)
            out[t] = torch.einsum("hvk,hk->hv", M, q[t][kidx]) * scale
            dest[int(indices[t])] = M
            records[int(record_indices[t]), :, i, :V] = dc
            records[int(record_indices[t]), :, i, V:V + K] = kk
            records[int(record_indices[t]), :, i, V + K] = a.view(HV)
        off += L
    return out, dest, records


def build_case(lengths, H, HV, seed=3, acc=None):
    """One recover case on CPU.  Write slots and record rows are the first
    T = sum(lengths) rows; the source pool has 3 trailing rows that must stay
    untouched."""
    T = sum(lengths)
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(T, H, K, generator=gen)
    k = torch.randn(T, H, K, generator=gen)
    q = q / q.norm(dim=-1, keepdim=True)
    k = k / k.norm(dim=-1, keepdim=True)
    v = torch.randn(T, HV, V, generator=gen)
    g = -torch.rand(T, HV, generator=gen) * 0.5
    beta = torch.rand(T, HV, generator=gen)
    S = T + 3
    source = torch.randn(S, HV, V, K, generator=gen) * 0.2
    indices = list(range(T))
    record_indices = list(range(T))
    scale = K ** -0.5
    ref_out, ref_dest, ref_records = reference_recover(
        q.to(torch.bfloat16).double(), k.to(torch.bfloat16).double(),
        v.to(torch.bfloat16).double(), g.double(),
        beta.to(torch.bfloat16).double(), source.double(), indices,
        record_indices, lengths, scale, dest_slots=S, accepted=acc)
    return {"lengths": lengths, "T": T, "H": H, "HV": HV, "S": S, "q": q,
            "k": k, "v": v, "g": g, "beta": beta, "source": source,
            "indices": indices, "record_indices": record_indices,
            "csl": [0, *lengths], "scale": scale, "accepted": acc,
            "ref_out": ref_out, "ref_dest": ref_dest,
            "ref_records": ref_records}


def to_dev(case, device):
    d = {"q": case["q"].to(torch.bfloat16).to(device),
         "k": case["k"].to(torch.bfloat16).to(device),
         "v": case["v"].to(torch.bfloat16).to(device),
         "g": case["g"].float().to(device),
         "beta": case["beta"].to(torch.bfloat16).to(device),
         "source": case["source"].float().to(device).contiguous(),
         "state": torch.full((case["S"], case["HV"], V, K), SENTINEL,
                             dtype=torch.float32, device=device),
         "csl": torch.tensor(case["csl"], dtype=torch.int32, device=device),
         "idx": torch.tensor(case["indices"], dtype=torch.int32, device=device),
         "ridx": torch.tensor(case["record_indices"], dtype=torch.int32,
                              device=device),
         "records": torch.full(
             (case["T"] + 2, case["HV"], max(max(case["lengths"]), MAX_MTP),
              V + K + 1), SENTINEL, dtype=torch.float32, device=device),
         "acc": None if case["accepted"] is None else torch.tensor(
             [case["accepted"]] * len(case["lengths"]), dtype=torch.int32,
             device=device),
         "scale": case["scale"]}
    return d


def run_stable(case, d):
    out = npu_recurrent_gated_delta_rule_recover(
        d["q"], d["k"], d["v"], d["state"], beta=d["beta"], scale=case["scale"],
        actual_seq_lengths=d["csl"], ssm_state_indices=d["idx"],
        num_accepted_tokens=d["acc"], g=d["g"], source_state=d["source"],
        record_indices=d["ridx"], records=d["records"])
    torch.npu.synchronize()
    return out


def run_aclnn(case, d):
    from fla_npu.ops.ascendc._runtime import call_aclnn

    out = torch.empty_like(d["v"])

    def build(ctx):
        return [ctx.tensor(d["q"], "query"), ctx.tensor(d["k"], "key"),
                ctx.tensor(d["v"], "value"), ctx.tensor(d["beta"], "beta"),
                ctx.tensor(d["source"], "source_state"),
                ctx.tensor(d["state"], "state"),
                ctx.tensor(d["csl"], "actual_seq_lengths"),
                ctx.tensor(d["idx"], "ssm_state_indices"),
                ctx.tensor(d["ridx"], "record_indices"),
                ctx.tensor(d["g"], "g"), ctx.tensor(None, "gk"),
                ctx.tensor(d["acc"], "num_accepted_tokens"),
                ctypes.c_float(case["scale"]), ctx.tensor(out, "out"),
                ctx.tensor(d["records"], "records")]

    call_aclnn("aclnnRecurrentGatedDeltaRuleRecover", build, (out,))
    torch.npu.synchronize()
    return out


def check_numeric(name, case, d, out):
    ok = True
    rel_out = rel(out.cpu(), case["ref_out"])
    ok &= rel_out < TOL_OUT
    dest = d["state"].cpu()
    written = sorted(set(case["indices"]))
    rel_dest = max(rel(dest[i], case["ref_dest"][i]) for i in written)
    ok &= rel_dest < TOL_STATE
    untouched = [i for i in range(case["S"]) if i not in written]
    dest_untouched = bool((dest[untouched] == SENTINEL).all().item())
    ok &= dest_untouched
    source_bit = bool((d["source"].cpu() == case["source"].float()).all().item())
    ok &= source_bit

    rec_full = d["records"].cpu()
    ref = case["ref_records"]
    rec = rec_full[:ref.shape[0], :, :ref.shape[2], :]
    cell_mask = ~torch.isnan(ref[..., V + K])
    rel_dc = rel(rec[..., :V][cell_mask], ref[..., :V][cell_mask])
    ok &= rel_dc < TOL_DC
    kidx = [hv // (case["HV"] // case["H"]) for hv in range(case["HV"])]
    kb = case["k"].to(torch.bfloat16)
    kref = torch.full_like(ref[..., V:V + K], float("nan"))
    off = 0
    for L in case["lengths"]:
        for i in range(L):
            kref[off + i, :, i, :] = kb[off + i][kidx].double()
        off += L
    mask = ~torch.isnan(kref)
    key_exact = bool((rec[..., V:V + K][mask] == kref[mask].float()).all().item())
    ok &= key_exact
    rel_alpha = rel(rec[..., V + K][cell_mask], ref[..., V + K][cell_mask])
    ok &= rel_alpha < TOL_ALPHA
    addr = sorted(set(case["record_indices"]))
    rows_ok = all(bool((rec_full[r] == SENTINEL).all())
                  for r in range(rec_full.shape[0]) if r not in addr)
    steps_ok = True
    off = 0
    for L in case["lengths"]:
        for i in range(L):
            t = off + i
            for j in range(rec_full.shape[2]):
                if j != i:
                    steps_ok &= bool((rec_full[t, :, j, :] == SENTINEL).all())
        off += L
    ok &= rows_ok and steps_ok
    report(name, ok,
           f"rel_out={rel_out:.3e} rel_dest={rel_dest:.3e} rel_dc={rel_dc:.3e} "
           f"key_exact={key_exact} rel_alpha={rel_alpha:.3e} "
           f"source_bit={source_bit} dest_untouched={dest_untouched} "
           f"records_untouched={rows_ok and steps_ok}")


def expect_reject(name, fn, needle=None):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        ok = needle is None or needle in str(exc)
        report(name, ok, f"{type(exc).__name__}: {str(exc)[:120]}")
        return
    report(name, False, "call was NOT rejected")


def scenario_numeric(device):
    for ci, lengths in enumerate(([4, 4, 4, 4], [1], [4, 4])):
        accepted = 2 if ci == 2 else None
        case = build_case(lengths, 16, 32, seed=11 + ci, acc=accepted)
        tag = f"B{len(lengths)}/L{lengths[0]}" + (f"/acc{accepted}" if accepted else "")
        for path, runner in (("stable", run_stable), ("aclnn", run_aclnn)):
            d = to_dev(case, device)
            check_numeric(f"numeric {tag} [{path}]", case, d, runner(case, d))
            del d
            torch.npu.empty_cache()


def scenario_rejections(device):
    case = build_case([4], 16, 32, seed=31)
    d = to_dev(case, device)

    def stable(**over):
        kw = dict(beta=d["beta"], scale=d["scale"], actual_seq_lengths=d["csl"],
                  ssm_state_indices=d["idx"], g=d["g"],
                  source_state=d["source"], record_indices=d["ridx"],
                  records=d["records"])
        kw.update(over)
        return npu_recurrent_gated_delta_rule_recover(
            d["q"], d["k"], d["v"], d["state"], **kw)

    # V == K == 128: transposing the inner pair keeps the shape but flips the
    # inner strides; both entries must reject it instead of reading the wrong
    # matrix back.
    src_t = d["source"].transpose(-1, -2)
    assert src_t.shape == d["source"].shape and src_t.stride(-1) != 1
    expect_reject("reject stable: source_state non-contiguous inner dims",
                  lambda: stable(source_state=src_t),
                  "last two dims must be contiguous")
    small_records = torch.zeros((case["T"], case["HV"], 1, V + K + 1),
                                dtype=torch.float32, device=device)
    expect_reject("reject stable: records dim 2 = 1 < MAX_MTP=8",
                  lambda: stable(records=small_records),
                  "must be >= MAX_MTP=8")
    expect_reject("reject stable: source_state IS the destination state",
                  lambda: stable(source_state=d["state"]),
                  "shares storage with the destination state")
    pool = torch.zeros((case["S"] + 1, case["HV"], V, K), dtype=torch.float32,
                       device=device)
    expect_reject("reject stable: source_state overlaps destination storage",
                  lambda: npu_recurrent_gated_delta_rule_recover(
                      d["q"], d["k"], d["v"], pool[1:], beta=d["beta"],
                      scale=d["scale"], actual_seq_lengths=d["csl"],
                      ssm_state_indices=d["idx"], g=d["g"],
                      source_state=pool[:-1], record_indices=d["ridx"],
                      records=d["records"]),
                  "shares storage with the destination state")
    gk = torch.zeros(case["T"], case["HV"], K, dtype=torch.float32,
                     device=device)
    expect_reject("reject stable: gk + records pairing",
                  lambda: stable(g=None, gk=gk), "not supported together with gk")
    expect_reject("reject stable: record_indices without records",
                  lambda: stable(record_indices=d["ridx"], records=None),
                  "must be provided together")

    # Control: disjoint slices of ONE storage are not an alias and must run.
    pool = torch.zeros((2 * case["S"], case["HV"], V, K), dtype=torch.float32,
                       device=device)
    src, dst = pool[:case["S"]], pool[case["S"]:]
    src.copy_(d["source"])
    dst.fill_(SENTINEL)
    out = npu_recurrent_gated_delta_rule_recover(
        d["q"], d["k"], d["v"], dst, beta=d["beta"], scale=d["scale"],
        actual_seq_lengths=d["csl"], ssm_state_indices=d["idx"], g=d["g"],
        source_state=src, record_indices=d["ridx"], records=d["records"])
    torch.npu.synchronize()
    rel_out = rel(out.cpu(), case["ref_out"])
    report("control stable: disjoint slices of one storage accepted",
           rel_out < TOL_OUT, f"rel_out={rel_out:.3e}")

    # The aclnn entry enforces the same layout/capacity rules (R1-1/R1-2).
    from fla_npu.ops.ascendc._runtime import call_aclnn

    def aclnn_with(source, records):
        out = torch.empty_like(d["v"])

        def build(ctx):
            return [ctx.tensor(d["q"]), ctx.tensor(d["k"]), ctx.tensor(d["v"]),
                    ctx.tensor(d["beta"]), ctx.tensor(source),
                    ctx.tensor(d["state"]), ctx.tensor(d["csl"]),
                    ctx.tensor(d["idx"]), ctx.tensor(d["ridx"]),
                    ctx.tensor(d["g"]), ctx.tensor(None), ctx.tensor(None),
                    ctypes.c_float(d["scale"]), ctx.tensor(out),
                    ctx.tensor(records)]

        call_aclnn("aclnnRecurrentGatedDeltaRuleRecover", build, (out,))
        torch.npu.synchronize()

    expect_reject("reject aclnn: source_state non-contiguous inner dims",
                  lambda: aclnn_with(src_t, d["records"]))
    expect_reject("reject aclnn: records dim 2 = 1 < MAX_MTP=8",
                  lambda: aclnn_with(d["source"], small_records))


def scenario_mutation_contract(device):
    case = build_case([2], 2, 4, seed=47)
    d = to_dev(case, device)
    v_state, v_records = d["state"]._version, d["records"]._version
    run_stable(case, d)
    report("mutation: one call bumps state and records exactly once",
           d["state"]._version == v_state + 1
           and d["records"]._version == v_records + 1,
           f"state +{d['state']._version - v_state} "
           f"records +{d['records']._version - v_records}")

    state2 = torch.full((case["S"], case["HV"], V, K), SENTINEL,
                        dtype=torch.float32, device=device)
    v2 = state2._version
    npu_recurrent_gated_delta_rule_recover(
        d["q"], d["k"], d["v"], state2, beta=d["beta"], scale=d["scale"],
        actual_seq_lengths=d["csl"], ssm_state_indices=d["idx"], g=d["g"])
    torch.npu.synchronize()
    report("mutation: call without records bumps state exactly once",
           state2._version == v2 + 1, f"state +{state2._version - v2}")

    grad_records = torch.zeros((case["T"], case["HV"], MAX_MTP, V + K + 1),
                               dtype=torch.float32, device=device,
                               requires_grad=True)
    expect_reject("mutation: records with requires_grad refused",
                  lambda: npu_recurrent_gated_delta_rule_recover(
                      d["q"], d["k"], d["v"], d["state"], beta=d["beta"],
                      scale=d["scale"], actual_seq_lengths=d["csl"],
                      ssm_state_indices=d["idx"], g=d["g"],
                      record_indices=d["ridx"], records=grad_records),
                  "must not require gradients")
    grad_state = d["state"].clone().detach().requires_grad_(True)
    expect_reject("mutation: state with requires_grad refused",
                  lambda: npu_recurrent_gated_delta_rule_recover(
                      d["q"], d["k"], d["v"], grad_state, beta=d["beta"],
                      scale=d["scale"], actual_seq_lengths=d["csl"],
                      ssm_state_indices=d["idx"], g=d["g"]),
                  "must not require gradients")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    torch.manual_seed(20261022)
    device = f"npu:{args.device}"

    scenario_numeric(device)
    scenario_rejections(device)
    scenario_mutation_contract(device)
    print("NOT_APPLICABLE primitive-graph replay: no graph-capture facility "
          "in this harness for this op family")
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} case(s): {FAILURES}")
        raise SystemExit(1)
    print("ALL PASS: recurrent_gated_delta_rule recover regression")


if __name__ == "__main__":
    main()

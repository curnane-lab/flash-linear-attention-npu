# GDN Recover Alias Contract v1

This contract governs how an engine may batch requests through the GDN state
pool(s) for one verify call — the base op
(`aclnnRecurrentGatedDeltaRule` / `fla_npu.ops.ascendc.npu_recurrent_gated_delta_rule`,
one mutated `state` operand) and the recover extension
(`aclnnRecurrentGatedDeltaRuleRecover` /
`fla_npu.ops.ascendc.npu_recurrent_gated_delta_rule_recover`, separate read-only
`source_state` and written `state`, plus the independent `records` output).

Each rule is labelled **[MEASURED]** (device behaviour observed on Ascend 910B3,
CANN 9.1.0) or **[ASSUMED]** (a guarantee the engine must supply; needs engine
source/scheduling evidence the device-side experiments cannot produce).

## Layer responsibility split

### The public entry guarantees (validated on host, no device sync)

The recover entries validate, from host-side metadata only:

- `record_indices` and `records` are provided together; `gk` (per-K gate) is
  rejected together with `records` (the alpha record field is scalar);
- `records` dim 2 (per-request step capacity) is >= `MAX_MTP = 8`, and its last
  dim equals `V + K + 1`;
- `source_state`'s last two dims are contiguous (`stride(-1) == 1`,
  `stride(-2) == K`) — the kernel reads it as contiguous V×K blocks;
- at the Stable-ABI python entry, `source_state` whose storage byte range
  overlaps the destination `state`'s is rejected (shape/stride/data_ptr
  metadata only, no sync);
- `state` and `records` are mutated in place: each version counter is bumped
  exactly once per call, and grad-requiring tensors are refused.

The host **never reads device tensor values**. Index-value validity
(`ssm_state_indices`, `record_indices` ranges) and per-request length limits
are therefore NOT checked by the op: out-of-range indices fault the device.

### The adapter must enforce (per-call batch construction)

- **Per-request disjoint write slots**: the per-token `ssm_state_indices`
  expansion must give distinct requests disjoint destination rows (rule 2).
- **Cross-request source/destination conflict rejection**: a batch in which one
  request's read row is a row another request of the same call writes is a
  contract violation and must be rejected by the adapter ahead of the public
  entry (rule 3). The host cannot see index values, so this check can only
  live in the adapter.
- Within ONE request, repeated slots are legal (same core, sequential, last
  write wins); only cross-request overlaps are rejected. Zero-length requests
  are total no-ops (no read, no write) and are skipped by the check.

### The engine / allocator owns

- **COW / write ownership of published pages** (rule 4, [ASSUMED]): once a
  state page has been published to other consumers (prefix sharing), the engine
  must treat it as read-only for verify paths and copy-on-write (or hold
  exclusive write ownership) for any mutation.
- **Record-row leases**: the allocation, ownership and lifetime of `records`
  rows across calls — which rows a call may write and when they are reclaimed —
  are allocator state, not op state.

## The four rules

### Rule 1 — multiple requests may read the SAME read-only source — [MEASURED at page level / CONSTRAINED at row level]

- Page level: several requests may read rows of one shared, read-only source
  allocation. Measured: 4 requests read rows of a single `source_state` page;
  the page is bitwise unchanged after the call and every request's output
  matches the fp64 snapshot oracle. Two concurrent calls on separate streams
  sharing one read-only source page likewise leave it bitwise unchanged.
- Row level: several requests may read the same *row* only if no request in the
  call writes it. With the base op, the read row of request *b* is also a write
  slot of *b* (index coupling), so a shared read row is necessarily written by
  each reader — it falls under rule 2 and is rejected. The row-level sharing
  form is only expressible in the recover namespace (separate read/write
  indices), where the shared row must be declared read-only.

### Rule 2 — destination write slots are mutually non-overlapping — [MEASURED]

When two requests of one call write the same pool slot, the slot's final
content is a **torn, schedule-dependent value** — not the last writer's, not
reproducible (contested-slot norms differ across consecutive calls). Therefore
the per-token `ssm_state_indices` expansion MUST give distinct requests
disjoint write slots, and the adapter must reject any batch with intersecting
per-request write sets.

### Rule 3 — cross-request source/destination cross-dependencies are EXPLICITLY REJECTED by the adapter — [MEASURED behaviour + enforcement]

Measured behaviour: the op does NOT order one request's read against another
request's write within a call. Where the read row is contended, the read races
with the earlier request's write: measured outcomes are pre-call (snapshot)
values, deterministically torn values (a fixed mix of pre-call and in-flight
write), or varying mixtures. Uncontended reads are always the pre-call value.
A batch in which request *j*'s read row is a row request *i* (*i*<j) writes is
therefore UNDEFINED — *j* may see the pre-call value or a torn mix, and the row
is additionally torn by the double write (rule 2). Such batches are contract
violations and the adapter must reject them instead of relying on any implied
ordering.

### Rule 4 — a published shared page is protected by engine COW / write ownership — [ASSUMED]

Once a state page has been published to other consumers (prefix sharing), the
engine must treat it as read-only for verify paths and copy-on-write (or
exclusive write ownership) for any mutation. Device-side proxies only: a
read-only `source_state` is bitwise preserved under concurrent multi-stream
reads, and disjoint-slot concurrent writes to one pool do not corrupt each
other's rows. Whether the *engine* honours COW/ownership for published pages
requires engine allocator/scheduler evidence that does not yet exist. Until
then this rule is an ASSUMED guarantee, not a measured one.

## What is measured vs what is assumed

| claim | status |
|---|---|
| uncontended initial-state read is the pre-call value | MEASURED |
| contended read: pre-call value OR torn mix, never reliably chained | MEASURED |
| contested write slot tears, non-deterministic | MEASURED |
| zero-length request is a total no-op | MEASURED |
| concurrent calls, disjoint slots, one pool: no cross-corruption | MEASURED |
| shared read-only source page unchanged, incl. across streams | MEASURED |
| op launch is async; per-stream events suffice for read-back | MEASURED |
| op is capturable/replayable with caller-owned stable buffers | MEASURED |
| published shared page protected by engine COW / write ownership | ASSUMED |
| any global cross-program read/write ordering | NOT ASSERTED |

**Explicit non-claim:** no global cross-program read/write ordering is asserted
anywhere in this contract. Repeated identical outcomes in tests are not
evidence of an ordering guarantee; ordering must come from the engine's event /
stream dependencies and from rules 1–4, not from observed stability.

# Warp partition review at current main

This record supersedes the source and validation status in the [earlier October 3 update](2026-10-03-update.md). That update retains the original reproductions, rejected designs, and earlier exact-wheel evidence.

| Target | Exact source |
|---|---|
| Base | [`88771e23f88b3e515cdcf585c4675248101ce72e`](https://github.com/triton-lang/triton/tree/88771e23f88b3e515cdcf585c4675248101ce72e) |
| [Gather #11988](https://github.com/triton-lang/triton/pull/11988) | [`15d553eeacd1918897a8d2e5ef642c15862dd855`](https://github.com/1sgtpepper/triton/tree/15d553eeacd1918897a8d2e5ef642c15862dd855) |
| [AtomicRMW #11989](https://github.com/triton-lang/triton/pull/11989) | [`b34d84c4d20fa99931c00a724743aff08d5f104f`](https://github.com/1sgtpepper/triton/tree/b34d84c4d20fa99931c00a724743aff08d5f104f) |
| Resolved combination | [`3b24e16778e657e42644abebb207ce9753ac6874`](https://github.com/1sgtpepper/triton/tree/3b24e16778e657e42644abebb207ce9753ac6874) |

Both PR histories were rebased onto the pinned base. `git range-diff` maps every original PR commit to a patch-identical rebased commit; each branch then adds one cleanup commit. Gather deletes a function whose Gather result was unused, together with its checks. The remaining observed epilogue-Gather and sibling-loop fixtures still cover forward rejection. Both branches remove an inherited `// Clear async_task.` comment that falsely described the ordinary retry path. Neither cleanup changes production control flow. The combination was re-resolved against the previously tested integration: its shared partitioner source and shared lit fixture are byte-identical to that integration. Relative to that prior integration, the two PR-owned files changed by the cleanup have only 44 deleted lines.

The newer base changes Gather's layout interface, Hopper memory lowering, and Hopper assembler selection, so old wheels are not evidence for these heads. The pinned validation workflows check each exact SHA before building.

## Exact-head compiler and packaging checks

[Expanded fork CI for both PR heads](https://github.com/1sgtpepper/triton/actions/runs/37094041633) passed its Gather and Atomic jobs. Each job checked out and asserted the exact PR SHA in the table, ran `pre-commit` from the pinned base, built the compiler, compiled its issue reproducer, ran all three relevant Hopper lit files (including the caller-owned task-ID diagnostic), and packaged a wheel. The [earlier exact-head run](https://github.com/1sgtpepper/triton/actions/runs/37089945002) also passed but ran only two lit files per PR; its wheels are the ones used for the H100 checks below. The [resolved-combination run](https://github.com/1sgtpepper/triton/actions/runs/37089951821) passed the same revision assertion, hooks, compiler build, both issue reproducers, all nine Hopper warp-specialization lit files, and wheel packaging. These are fork runs; the upstream PRs have no required CI result from them.

| H100-tested wheel from the earlier head run or combination run | SHA-256 of downloaded wheel | GitHub artifact ZIP SHA-256 |
|---|---|---|
| Gather `3.9.0+git15d553ee` | `eeada6c07c976682db6fdcfdb9eb82406af77250d73a053df5eef58e3bc441f4` | `d13b8f3cba51ef46129ae7e34cf522ec84b3900c903eceae118752e7de88492f` |
| Atomic `3.9.0+gitb34d84c4` | `b39b96782fab3a0cd678fda09e84bca3f7f4b61a7a597268c97d13282bf8b4f5` | `0f8b6c9bdb771fa8b01e28fa63b28dbc6fa559713cb7fba1baf33b8add37ea05` |
| Combination `3.9.0+git3b24e167` | `5ecbd2cf7b24513ae7f10a265814faf33704b62b6dbece73ae0a15f800ec507f` | `09cb342360da893b6d097b3b6a3bbdb71036dac8e80491e52a9006e9c3eb0600` |

## H100 execution on those exact wheels

The [runner](run_hopper.py) checked each mounted wheel and test-file checksum before running one isolated pytest case on an NVIDIA H100 80GB HBM3 (driver 580.95.05, compute capability 9.0). Every case below reports `exit_status: 0` and `1 passed`; raw JSON preserves the checksums, stdout, stderr, and device. The four isolated repository test-file runs emitted unknown-marker warnings because the repository's pytest marker configuration was not mounted. Those warnings did not skip the selected tests.

| Exact wheel | Numerical or effect case | H100 run | Raw result |
|---|---|---|---|
| Gather | [Retained supported partition](https://github.com/1sgtpepper/triton/blob/15d553eeacd1918897a8d2e5ef642c15862dd855/python/test/unit/language/test_warp_specialization.py#L105-L126) | [passed](https://modal.com/apps/sgtpepper/main/ap-fHr7v68yj1qobJnT4fDx4e) | [JSON](results/rebased-gather-positive.json) |
| Gather | [Unsupported Gather fallback compared with plain execution and reference](https://github.com/1sgtpepper/triton/blob/15d553eeacd1918897a8d2e5ef642c15862dd855/python/test/unit/language/test_warp_specialization.py#L61-L101) | [passed](https://modal.com/apps/sgtpepper/main/ap-65Wl76AraSdW8K4RoZFihJ) | [JSON](results/rebased-gather-fallback.json) |
| Atomic | [Scalar producer return value and atomic effect](https://github.com/1sgtpepper/triton/blob/b34d84c4d20fa99931c00a724743aff08d5f104f/python/test/unit/language/test_warp_specialization.py#L361-L385) | [passed](https://modal.com/apps/sgtpepper/main/ap-SZTW6VGJfQGk8w4iZnBFdG) | [JSON](results/rebased-atomic-positive.json) |
| Atomic | [Rejected partition preserves atomic effects](https://github.com/1sgtpepper/triton/blob/b34d84c4d20fa99931c00a724743aff08d5f104f/python/test/unit/language/test_warp_specialization.py#L310-L357) | [passed](https://modal.com/apps/sgtpepper/main/ap-MyorMrPSSrcEwuOhQu09Me) | [JSON](results/rebased-atomic-fallback.json) |
| Combination | [Both operations in a supported partition](test_gather_atomic_positive.py) | [passed](https://modal.com/apps/sgtpepper/main/ap-kVzjd7pdc1DI5MXOQjvRVx) | [JSON](results/rebased-mixed-positive.json) |
| Combination | [Both operations falling back](test_gather_atomic_fallback.py) | [passed](https://modal.com/apps/sgtpepper/main/ap-Q1MBkQ1don54gV0ROW9D4z) | [JSON](results/rebased-mixed-fallback.json) |

## Remaining review decisions

The ordinary three-to-two-warp-group `Retry` path can retain generated task IDs; [#10436](https://github.com/triton-lang/triton/pull/10436) owns that independent issue. These PRs use `Unsupported` to exit before retrying. A cleanup failure after structural rewriting is terminal by source control flow, but no deterministic valid fixture has forced that branch. The [Gather maintainer's architecture objection](https://github.com/triton-lang/triton/pull/11988#pullrequestreview-5347572088) still requires human review; compiler and numerical results do not resolve it by themselves.

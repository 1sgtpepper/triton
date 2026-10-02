# Warp-specialization partition recovery: remote review packet

**Current review:** [October 3 source and validation update](2026-10-03-update.md). The status and revision table below record the earlier October 2 snapshot.

## Verdict and provenance

The bounded Gather and AtomicRMW fallbacks are validated for the reported kernels and listed compiler fixtures. The two upstream pull requests are not ready for unconditional sign-off: the Gather review still requests changes, and a deterministic execution test for the exceptional post-rewrite cleanup failure has not been constructed.

| Source | Immutable revision or run | Purpose |
|---|---|---|
| [Gather PR #11988](https://github.com/triton-lang/triton/pull/11988) | `d12cd27be96c2a9c7fc387ba619672c4aaccb4a8` | Corrected Gather PR head; draft, changes requested |
| [AtomicRMW PR #11989](https://github.com/triton-lang/triton/pull/11989) | `6c1cbfe44f30c53aaae155299e67a62bc71403d4` | Corrected Atomic PR head; draft, review required |
| Common PR base | `895801b66e43eb6784624d9e68870af3b8fd78d8` | Base for both PRs |
| [Integrated corrected code snapshot](https://github.com/1sgtpepper/triton/tree/06019465ecf45c6786c635cb4c62a436b6cd4af6) | `06019465ecf45c6786c635cb4c62a436b6cd4af6` | Gather prior head plus resolved Atomic overlay and corrected fixture |
| [Fork compiler validation](https://github.com/1sgtpepper/triton/actions/runs/36973035391) | Workflow revision `522ed82f4b413b8e03bfc84fda638614ea3c471f` | Exact patch application, builds, issue compile tests, Hopper lit, wheels |

The corrected snapshot's code diff against the prior Gather PR head 4f12ec924a55271fdd29617c97660446c2263ef6 has SHA-256 62de4dd9923ef1a2e8df290790892bdc8bbdbc16e1c930d5533123a89bfd8783. It exactly matches the [integration patch](https://github.com/1sgtpepper/triton/blob/522ed82f4b413b8e03bfc84fda638614ea3c471f/validation/patches/integration.patch) applied by fork CI. The [Gather correction](https://github.com/1sgtpepper/triton/blob/522ed82f4b413b8e03bfc84fda638614ea3c471f/validation/patches/gather.patch) and [Atomic correction](https://github.com/1sgtpepper/triton/blob/522ed82f4b413b8e03bfc84fda638614ea3c471f/validation/patches/atomic.patch) match the new individual PR commits byte for byte; their SHA-256 values are 9eb5a7325daa72666ca6b0718ef65c648ce4083f10585f0364c7a092530a23b6 and 0448ee8d24a30b4dffc68451329d43336b6a665943d800ab41642fc8102ca5ec. The Atomic patch applies to its prior PR head 33fc7879995fef07a6b64415482c63e936887f47.

Both corrected commits are now on their upstream draft PR branches. The integrated snapshot is a separate review view of both changes. The validation workflow checked out the pinned prior heads and applied the exact patch files; the snapshot itself was not its checkout.

## Behavior contract and owner

An unsupported selected warp-specialization partition may return to ordinary staged execution only while the function has not been structurally rewritten; a cleanup failure after rewriting must terminate the pass.

The [driver](https://github.com/1sgtpepper/triton/blob/06019465ecf45c6786c635cb4c62a436b6cd4af6/third_party/nvidia/hopper/lib/Transforms/WarpSpecialization.cpp#L47-L112) records preexisting task IDs, runs task partition and propagation, consumes the data-partition result, and owns the fallback. The [partitioner](https://github.com/1sgtpepper/triton/blob/06019465ecf45c6786c635cb4c62a436b6cd4af6/third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSDataPartition.cpp#L1377-L1451) chooses candidate dimensions and owns structural rewriting. The shared [result type](https://github.com/1sgtpepper/triton/blob/06019465ecf45c6786c635cb4c62a436b6cd4af6/third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSDataPartition.h#L8-L18) now distinguishes `Success`, pre-rewrite `Retry`, pre-rewrite `Unsupported`, and terminal `FailedAfterRewrite`.

The cross-attempt proof is source-level. [Task partition](https://github.com/1sgtpepper/triton/blob/06019465ecf45c6786c635cb4c62a436b6cd4af6/third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSTaskPartition.cpp#L20-L117) assigns attributes; [task-ID propagation](https://github.com/1sgtpepper/triton/blob/06019465ecf45c6786c635cb4c62a436b6cd4af6/third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSTaskIdPropagate.cpp#L26-L64) propagates attributes. Candidate calculation returns `Retry` or `Unsupported` before [rematerialization and slicing](https://github.com/1sgtpepper/triton/blob/06019465ecf45c6786c635cb4c62a436b6cd4af6/third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSDataPartition.cpp#L1377-L1428). [Deep cleanup](https://github.com/1sgtpepper/triton/blob/06019465ecf45c6786c635cb4c62a436b6cd4af6/third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSDataPartition.cpp#L1269-L1335) may erase operations before greedy canonicalization fails; its failure now returns `FailedAfterRewrite`, which the driver treats as pass failure rather than another attempt or attribute-only recovery. Preexisting caller task IDs also prevent successful `Unsupported` fallback. No test currently forces the rare cleanup-failure return.

## Review findings and corrections

1. **Matched positive Gather experiment.** An earlier positive candidate failed an attribute-free `ASTSource` compilation on a two-byte async copy. That did not reproduce normal JIT pointer specialization. The [matched JIT probe](test_gather_matched_jit.py) captured three aligned Torch pointers, `tt.divisibility=16` on each, CUDA architecture 90, and `num_stages=3`. With the [positive candidate wheel](https://github.com/1sgtpepper/triton/actions/runs/36830582609), warmup compilation succeeded and TTGIR contained both Gather and `ttg.warp_specialize`. Actual launch failed before numerical comparison: 262400 bytes of shared memory required, H100 limit 232448. See the raw [positive result](results/positive-candidate.json) and [run](https://modal.com/apps/sgtpepper/main/ap-36AYSxOCBLVxL0sODlkUdq). This rejects that candidate for the reported launch settings, not all possible positive designs. The matched [fallback control](results/matched-fallback.json) compiled, retained Gather, removed specialization, and passed the numerical reference on [H100](https://modal.com/apps/sgtpepper/main/ap-YBeb73hOpFwAI39Lubo6Dn).
2. **Lost alternate-candidate fixture.** The ninth Gather full-pass descriptor fixture crashed for reasons independently observable without Gather. It was removed from [ws_gather_fallback.mlir](../../test/Hopper/WarpSpecialization/ws_gather_fallback.mlir). Its useful rejected-M/accepted-N assertion was restored at the owning pass in [ws_data_partition.mlir, lines 50–89](../../test/Hopper/WarpSpecialization/ws_data_partition.mlir). The fixture binds one retained Gather to the A allocation used by both dots, checks two N-half B loads, and excludes a duplicate Gather across the function. This direct-pass control avoids the unrelated descriptor and async-token path.
3. **Post-rewrite retry.** The previous cleanup failure returned `Retry` after slicing. The terminal result and driver case described above close that source-path gap. An executed deterministic failure witness remains missing.
4. **BF16 zero-bias atomic.** The previous K-partial case was restored as an expected rejection in [ws_atomic_rmw_partition_reject.mlir](../../test/Hopper/WarpSpecialization/ws_atomic_rmw_partition_reject.mlir). [This executable arithmetic witness](bf16_partial_rounding.py) gives FP32 partials `1 + 1/256` and `-1`: one BF16 conversion after summing gives `1/256`, while BF16 conversion of each partial before summing gives zero. This is arithmetic evidence against that rewrite, not a reproduced H100 miscompilation.
5. **Masked load `other` attribution.** [WSLowerMem.cpp](../../third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSLowerMem.cpp) passes `other` into the async-copy op; the later NVIDIA conversion in [LoadStoreOpToLLVM.cpp](../../third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp) assumes zero rather than implementing arbitrary `other`. Positive support must address the downstream contract. It is outside these fallback patches.

The bounded fallback avoids a generic Gather blacklist, optimistic pointer-alignment assumptions, atomic task seeding, and attribute cleanup presented as structural rollback. Positive transport would additionally require proof of producer completion, legal async-copy width, masked values, synchronization, and launch resources.

## Validation ledger

| Evidence | Exact result |
|---|---|
| [Final fork CI](https://github.com/1sgtpepper/triton/actions/runs/36973035391) | All three jobs passed. Gather: one issue compile test and two lit files, including the new direct fixture. Atomic: one issue compile test and two lit files. Integration: both issue compile tests and all **nine** `test/Hopper/WarpSpecialization` lit files. [Selected output lines](ci-final-summary.txt) preserve the exact test counts and file names; the Actions run has the full log. |
| Corrected Gather-only wheel | [H100 numerical test passed](https://modal.com/apps/sgtpepper/main/ap-bCBzpQEe6UxmCKqfnfqVZ1); [raw result](results/gather-only.json); wheel SHA-256 `b3a61020628184b3a5613725b7765017098e09873f864a2e28f81c16211c7eda`. |
| Corrected Atomic-only wheel | [H100 output and separate atomic-effect test passed](https://modal.com/apps/sgtpepper/main/ap-nzowK7UMoWOWLzQRz1LyZQ); [raw result](results/atomic-only.json); wheel SHA-256 `4b3208db5a7ec54eb8037e994c5a6b83ca0ef8ddc7d98ef10311a949d3fc8f5f`. |
| Corrected integration wheel | [Gather](https://modal.com/apps/sgtpepper/main/ap-T6bOpTxQFiZDZOxemvz8ah) and [Atomic](https://modal.com/apps/sgtpepper/main/ap-XDTdcaBA4W5px4tngDk3S0) H100 numerical tests passed on the **same wheel**; [Gather result](results/integration-gather.json), [Atomic result](results/integration-atomic.json); wheel SHA-256 `f8f0325ee7259b5676bcacab249adddbba67cbfcf4543e434af8f6ceaf9cca5e`. |
| Positive candidate | [Fork wheel build/package passed](https://github.com/1sgtpepper/triton/actions/runs/36830582609); matched JIT compiled specialization but [launch failed](results/positive-candidate.json) on shared-memory capacity. Wheel SHA-256 `1428391dce42c34ef87c0aaa549997404b60891df8f0bb17981342db871c2dbc`. |
| Static checks | Exact patches passed `git apply --check` against the pinned heads; base-to-head `git diff --check` passed; pinned clang-format 19.1.6 dry run passed for changed C++ files. |

The [October 2 fixture commit](https://github.com/1sgtpepper/triton/commit/06019465ecf45c6786c635cb4c62a436b6cd4af6) only strengthens the Gather lit assertion. The production compiler source is unchanged from the earlier H100-tested corrected wheels.

The [H100 runner](run_hopper.py) records device, wheel checksum, mounted-test checksum, pytest exit status, stdout, and stderr. The isolated pytest files emitted unknown-marker warnings because the repository marker configuration was not mounted; the listed numerical tests did execute. The GitHub Actions build logs and workflow at the pinned validation revision are the primary compiler validation record. The code snapshot itself was not the CI checkout; its code diff is byte-for-byte the tested integration patch.

## Remaining review obligations

- Find a deterministic, valid way to exercise cleanup failure *after* structural rewriting, if possible without a production test hook. Until then, the terminal branch has source-level control-flow proof rather than behavior-first execution proof.
- Reassess the current owner-local fallback against the Gather maintainer's still-open changes-requested review. The corrected evidence does not constitute maintainer acceptance.
- Do not infer numerical correctness of the positive candidate: its matched launch never ran. Do not infer general impossibility of positive support from either candidate's failure.
- Keep descriptor-coordinate and generic pointer-load token failures separate from these two reported fallback fixes.

The corrected code is on both draft PR heads and independently reviewable at the pinned integrated snapshot above.

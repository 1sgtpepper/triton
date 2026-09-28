import os
import resource
import subprocess
import sys

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def gather_before_dot(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                      WARP_SPECIALIZE: tl.constexpr):
    om = tl.arange(0, M)
    on = tl.arange(0, N)
    ok = tl.arange(0, K)
    acc = tl.zeros([M, N], dtype=tl.float32)
    for i in tl.range(0, 4, warp_specialize=WARP_SPECIALIZE):
        a = tl.load(a_ptr + om[:, None] * K + ok[None, :] + i * M * K)
        b = tl.load(b_ptr + ok[:, None] * N + on[None, :] + i * K * N)
        indices = tl.full([M, K], 1, tl.int32)
        a = tl.gather(a, indices, axis=1)
        acc += tl.dot(a, b)
    tl.store(c_ptr + om[:, None] * N + on[None, :], acc)


@triton.jit
def atomic_after_dot(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                     WARP_SPECIALIZE: tl.constexpr):
    om = tl.arange(0, M)
    on = tl.arange(0, N)
    ok = tl.arange(0, K)
    acc = tl.zeros([M, N], dtype=tl.float32)
    for i in tl.range(0, 4, warp_specialize=WARP_SPECIALIZE):
        a = tl.load(a_ptr + om[:, None] * K + ok[None, :] + i * M * K)
        b = tl.load(b_ptr + ok[:, None] * N + on[None, :] + i * K * N)
        acc += tl.dot(a, b)
        tl.atomic_add(c_ptr + om[:, None] * N + on[None, :], acc)
    tl.store(c_ptr + om[:, None] * N + on[None, :], acc)


def _compile_reproducer(issue, warp_specialize):
    if issue == "11952":
        fn = gather_before_dot
    elif issue == "11954":
        fn = atomic_after_dot
    else:
        raise ValueError(f"unknown issue: {issue}")

    source = ASTSource(
        fn=fn,
        signature={"a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": "*fp32"},
        constexprs={"M": 128, "N": 128, "K": 128, "WARP_SPECIALIZE": warp_specialize},
    )
    triton.compile(
        source,
        target=GPUTarget("cuda", 90, 32),
        options={"num_warps": 4, "num_stages": 3},
    )


def _disable_core_dumps():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _run_compiler(issue, warp_specialize):
    return subprocess.run(
        [sys.executable, __file__, "--compile", issue,
         str(warp_specialize)],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        preexec_fn=_disable_core_dumps,
        check=False,
        timeout=300,
    )


def _assert_main_fails_at(issue, expected):
    result = _run_compiler(issue, True)
    output = result.stdout + result.stderr
    assert result.returncode != 0, f"issue #{issue} unexpectedly compiled on main"
    assert expected in output, f"issue #{issue} failed for another reason:\n{output[-4000:]}"
    print(f"Reproduced issue #{issue}: {expected}")


def _assert_regular_pipeline_compiles(issue):
    result = _run_compiler(issue, False)
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"issue #{issue} failed without warp specialization:\n{output[-4000:]}"


def test_main_gather_reproducer():
    _assert_regular_pipeline_compiles("11952")
    _assert_main_fails_at("11952", "Unexpected op")


def test_main_atomic_reproducer():
    _assert_regular_pipeline_compiles("11954")
    _assert_main_fails_at("11954", "Unexpected asyncTaskIds.size()")


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "--compile":
    _compile_reproducer(sys.argv[2], sys.argv[3] == "True")

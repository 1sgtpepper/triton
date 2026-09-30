"""Compile the Gather issue shape and an eligible N-shard counterexample."""

import sys

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr,
           K: tl.constexpr, AXIS: tl.constexpr):
    om = tl.arange(0, M)
    on = tl.arange(0, N)
    ok = tl.arange(0, K)
    acc = tl.zeros([M, N], tl.float32)
    for i in tl.range(0, 4, warp_specialize=True):
        a = tl.load(a_ptr + om[:, None] * K + ok[None, :] + i * M * K)
        b = tl.load(b_ptr + ok[:, None] * N + on[None, :] + i * K * N)
        indices = tl.full([M, K], 1, tl.int32)
        acc += tl.dot(tl.gather(a, indices, axis=AXIS), b)
    tl.store(c_ptr + om[:, None] * N + on[None, :], acc)


if __name__ == "__main__":
    n = int(sys.argv[1])
    axis = int(sys.argv[2])
    source = ASTSource(
        fn=kernel,
        signature={"a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": "*fp32"},
        constexprs={"M": 128, "N": n, "K": 128, "AXIS": axis},
    )
    compiled = triton.compile(
        source, target=GPUTarget("cuda", 90, 32),
        options={"num_warps": 4, "num_stages": 3},
    )
    print("N:", n, "axis:", axis)
    print("warp_specialize retained:", "ttg.warp_specialize" in compiled.asm["ttgir"])

import json

import torch
import triton
import triton.language as tl
from triton.runtime.driver import driver


@triton.jit
def gather_before_dot(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    om = tl.arange(0, M)
    on = tl.arange(0, N)
    ok = tl.arange(0, K)
    acc = tl.zeros([M, N], dtype=tl.float32)
    for i in tl.range(0, 4, warp_specialize=True):
        a = tl.load(a_ptr + om[:, None] * K + ok[None, :] + i * M * K)
        b = tl.load(b_ptr + ok[:, None] * N + on[None, :] + i * K * N)
        idx = tl.full([M, K], 1, tl.int32)
        a = tl.gather(a, idx, axis=1)
        acc += tl.dot(a, b)
    tl.store(c_ptr + om[:, None] * N + on[None, :], acc)


def test_gather_matched_jit_specialization():
    M = N = K = 128
    torch.manual_seed(11952)
    a = torch.randn((4, M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((4, K, N), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float32)

    device = driver.active.get_current_device()
    _, _, target, backend, binder = gather_before_dot.device_caches[device]
    bound, specialization, options = binder(
        a, b, c, M, N, K, num_ctas=1, num_warps=4, num_stages=3
    )
    options, signature, constexprs, attrs = gather_before_dot._pack_args(
        backend, {"num_ctas": 1, "num_warps": 4, "num_stages": 3}, bound, specialization, options
    )
    evidence = {
        "target": str(target),
        "pointer_alignment_mod_16": [x.data_ptr() % 16 for x in (a, b, c)],
        "signature": signature,
        "attrs": {str(k): v for k, v in attrs.items()},
        "num_stages": options.num_stages,
    }
    print("MATCHED_JIT_SPECIALIZATION " + json.dumps(evidence, sort_keys=True))

    compiled = gather_before_dot.run(
        a, b, c, M, N, K, grid=(1,), warmup=True,
        num_ctas=1, num_warps=4, num_stages=3,
    )
    print("MATCHED_JIT_COMPILED " + json.dumps({
        "has_gather": "tt.gather" in compiled.asm["ttgir"],
        "has_warp_specialize": "ttg.warp_specialize" in compiled.asm["ttgir"],
    }, sort_keys=True))

    gather_before_dot[(1,)](a, b, c, M, N, K, num_ctas=1, num_warps=4, num_stages=3)
    expected = torch.zeros((M, N), device="cuda", dtype=torch.float32)
    for i in range(4):
        selected = a[i, :, 1:2].float().expand(M, K)
        expected += selected @ b[i].float()
    print("MATCHED_JIT_MAX_ABS_ERROR " + str((c - expected).abs().max().item()))
    torch.testing.assert_close(c, expected, atol=5e-2, rtol=5e-2)

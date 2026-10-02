import torch
import triton
import triton.language as tl


@triton.jit
def gather_atomic_mixed(a_ptr, b_ptr, out_ptr, atomic_ptr, WARP_SPECIALIZE: tl.constexpr):
    om = tl.arange(0, 128)
    on = tl.arange(0, 128)
    ok = tl.arange(0, 128)
    acc = tl.zeros([128, 128], dtype=tl.float32)
    for i in tl.range(0, 4, warp_specialize=WARP_SPECIALIZE):
        a = tl.load(a_ptr + i * 128 * 128 + om[:, None] * 128 + ok[None, :])
        b = tl.load(b_ptr + i * 128 * 128 + ok[:, None] * 128 + on[None, :])
        indices = tl.full([128, 128], 1, tl.int32)
        gathered = tl.gather(a, indices, axis=1)
        acc += tl.dot(gathered, b)
        tl.atomic_add(atomic_ptr + om[:, None] * 128 + on[None, :], acc)
    tl.store(out_ptr + om[:, None] * 128 + on[None, :], acc)


def test_gather_atomic_mixed_fallback():
    torch.manual_seed(1198811989)
    a = torch.randn((4, 128, 128), device="cuda", dtype=torch.float16)
    b = torch.randn((4, 128, 128), device="cuda", dtype=torch.float16)
    out_ws = torch.empty((128, 128), device="cuda", dtype=torch.float32)
    out_plain = torch.empty_like(out_ws)
    effect_ws = torch.zeros_like(out_ws)
    effect_plain = torch.zeros_like(out_ws)

    compiled = gather_atomic_mixed[(1, )](a, b, out_ws, effect_ws, True, num_warps=4, num_stages=3)
    gather_atomic_mixed[(1, )](a, b, out_plain, effect_plain, False, num_warps=4, num_stages=3)
    assert "tt.gather" in compiled.asm["ttgir"]
    assert "tt.atomic_rmw" in compiled.asm["ttgir"]
    assert "ttg.warp_specialize" not in compiled.asm["ttgir"]
    expected = torch.zeros_like(out_ws)
    expected_effect = torch.zeros_like(out_ws)
    for i in range(4):
        selected = a[i, :, 1:2].float().expand(128, 128)
        expected += selected @ b[i].float()
        expected_effect += expected
    torch.testing.assert_close(out_ws, expected, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out_ws, out_plain, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(effect_ws, expected_effect, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(effect_ws, effect_plain, atol=5e-2, rtol=5e-2)

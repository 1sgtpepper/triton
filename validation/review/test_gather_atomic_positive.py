import torch
import triton
import triton.language as tl


@triton.jit
def gather_atomic_positive(a_ptr, b_ptr, out_ptr, counter_ptr, offset_ptr, WARP_SPECIALIZE: tl.constexpr):
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[256, 32], strides=[32, 1], block_shape=[64, 32])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[128, 512], strides=[512, 1], block_shape=[32, 512])
    acc = tl.zeros([64, 512], dtype=tl.float32)
    for i in tl.range(0, 4, warp_specialize=WARP_SPECIALIZE):
        offset = tl.load(offset_ptr)
        previous = tl.atomic_add(counter_ptr, 32, sem="relaxed")
        a = a_desc.load([i * 64, 0])
        b = b_desc.load([offset + previous, 0])
        indices = tl.full([64, 32], 1, tl.int32)
        acc = tl.dot(tl.gather(a, indices, axis=1), b, acc)
    om = tl.arange(0, 64)
    on = tl.arange(0, 512)
    tl.store(out_ptr + om[:, None] * 512 + on[None, :], acc.to(tl.float16))


def test_gather_atomic_positive():
    torch.manual_seed(1198889)
    a = torch.randn((4, 64, 32), device="cuda", dtype=torch.float16)
    b = torch.randn((4, 32, 512), device="cuda", dtype=torch.float16)
    out_ws = torch.empty((64, 512), device="cuda", dtype=torch.float16)
    out_plain = torch.empty_like(out_ws)
    counter_ws = torch.zeros(1, device="cuda", dtype=torch.int32)
    counter_plain = torch.zeros_like(counter_ws)
    offset = torch.zeros_like(counter_ws)
    triton.set_allocator(lambda size, align, stream: torch.empty(size, device="cuda", dtype=torch.uint8))

    compiled = gather_atomic_positive[(1, )](a, b, out_ws, counter_ws, offset, True, num_warps=4, num_stages=3)
    gather_atomic_positive[(1, )](a, b, out_plain, counter_plain, offset, False, num_warps=4, num_stages=3)
    assert "tt.gather" in compiled.asm["ttgir"]
    assert "tt.atomic_rmw" in compiled.asm["ttgir"]
    assert "ttg.warp_specialize" in compiled.asm["ttgir"]
    assert counter_ws.item() == counter_plain.item() == 128
    expected = sum(a[i, :, 1:2].float().expand(64, 32) @ b[i].float() for i in range(4)).half()
    torch.testing.assert_close(out_ws, expected, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out_ws, out_plain, atol=5e-2, rtol=5e-2)

import triton
import triton.language as tl

@triton.jit
def _loop_dot_scaled(lhs, lhs_sf, rhs, rhs_sf, out, NBLK: tl.constexpr):
    rn = tl.arange(0, 128); rk2 = tl.arange(0, 64); rs = tl.arange(0, 4)
    for blk in tl.range(0, NBLK, num_stages=2):
        off = blk * 128
        a = tl.load(lhs + (off + rn)[:, None] * 64 + rk2[None, :]).to(tl.uint8)
        a_sf = tl.load(lhs_sf + (off + rn)[:, None] * 4 + rs[None, :])
        b = tl.load(rhs + rk2[:, None] * 64 * NBLK + (off + rn)[None, :]).to(tl.uint8)
        b_sf = tl.load(rhs_sf + (off + rn)[:, None] * 4 + rs[None, :])
        acc = tl.dot_scaled(a, a_sf, "e2m1", b, b_sf, "e2m1", out_dtype=tl.float32)
        tl.store(out + (off + rn)[:, None] * 128 + rn[None, :], acc)

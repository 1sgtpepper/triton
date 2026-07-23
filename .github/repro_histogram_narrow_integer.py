import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def histogram_kernel(x_ptr, out_ptr):
    values = tl.load(x_ptr + tl.arange(0, 256))
    histogram = tl.histogram(values, 4)
    tl.store(out_ptr + tl.arange(0, 4), histogram)


for element_ty in ("i8", "i16", "i32"):
    triton.compile(
        ASTSource(
            fn=histogram_kernel,
            signature={"x_ptr": f"*{element_ty}", "out_ptr": "*i32"},
            constexprs={},
        ),
        target=GPUTarget("cuda", 90, 32),
    )

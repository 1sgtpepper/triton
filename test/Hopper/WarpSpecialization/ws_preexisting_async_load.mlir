// RUN: not triton-opt %s --nvgpu-warp-specialization=num-stages=2 -o /dev/null 2>&1 | FileCheck %s

// CHECK: warp specialization cannot fall back from ordinary-load producer channel with preexisting async_task_id attributes

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
  tt.func @preexisting_ordinary_load_channel(%ptr: tensor<128x64x!tt.ptr<f16>, #blocked>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %c1 step %c1 {
      %load = tt.load %ptr {async_task_id = array<i32: 0>} : tensor<128x64x!tt.ptr<f16>, #blocked>
      %buffer = ttg.local_alloc %load {async_task_id = array<i32: 1, 2>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      scf.yield
    } {tt.num_stages = 2 : i32, tt.warp_specialize}
    tt.return
  }
}

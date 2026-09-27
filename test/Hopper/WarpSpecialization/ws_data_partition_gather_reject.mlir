// RUN: not --crash triton-opt %s --nvgpu-test-ws-data-partition=num-warp-groups=3 -o /dev/null

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @gather_on_partitioned_axis(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: i32, %arg3: i32) {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %c1 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i32
    %acc = arith.constant {async_task_id = array<i32: 1, 2>} dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %ptr_a = tt.splat %arg0 {async_task_id = array<i32: 0>} : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %ptr_b = tt.splat %arg1 {async_task_id = array<i32: 0>} : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #blocked1>
    %idx = arith.constant {async_task_id = array<i32: 0>} dense<1> : tensor<128x64xi32, #blocked>
    %result = scf.for %i = %c0 to %arg2 step %c1 iter_args(%carry = %acc) -> tensor<128x128xf32, #mma> {
      %a = tt.load %ptr_a {async_task_id = array<i32: 0>} : tensor<128x64x!tt.ptr<f16>, #blocked>
      %g = tt.gather %a[%idx] {async_task_id = array<i32: 0>, axis = 0 : i32} : (tensor<128x64xf16, #blocked>, tensor<128x64xi32, #blocked>) -> tensor<128x64xf16, #blocked>
      %a_smem = ttg.local_alloc %g {async_task_id = array<i32: 1, 2>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b = tt.load %ptr_b {async_task_id = array<i32: 0>} : tensor<64x128x!tt.ptr<f16>, #blocked1>
      %b_smem = ttg.local_alloc %b {async_task_id = array<i32: 1, 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %dot = ttng.warp_group_dot %a_smem, %b_smem, %carry {async_task_id = array<i32: 1, 2>, inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x128xf16, #shared, #smem> -> tensor<128x128xf32, #mma>
      scf.yield %dot : tensor<128x128xf32, #mma>
    }
    tt.return
  }
}

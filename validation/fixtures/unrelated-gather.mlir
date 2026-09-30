// RUN: triton-opt %s --nvgpu-warp-specialization=num-stages=2 -o /dev/null

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
  tt.func @unrelated_epilogue_gather_keeps_warp_specialization(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: !tt.ptr<f16>, %arg4: !tt.ptr<f16>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %c128 = arith.constant 128 : i32
    %c256 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %a_desc = tt.make_tensor_descriptor %arg0, [%c128, %c256], [%c256_i64, %c1_i64] : <f16>, <128x64xf16, #shared>
    %b_desc = tt.make_tensor_descriptor %arg1, [%c256, %c256], [%c256_i64, %c1_i64] : <f16>, <64x256xf16, #shared>
    %c_desc = tt.make_tensor_descriptor %arg2, [%c128, %c256], [%c256_i64, %c1_i64] : <f16>, <128x256xf16, #shared>
    %independent_desc = tt.make_tensor_descriptor %arg3, [%c128, %c256], [%c256_i64, %c1_i64] : <f16>, <128x256xf16, #shared>
    %init = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %acc = scf.for %i = %c0 to %c4 step %c1 iter_args(%iter = %init) -> tensor<128x256xf32, #mma> : i32 {
      %a = tt.descriptor_load %a_desc[%c0, %i] : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
      %a_smem = ttg.local_alloc %a : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b = tt.descriptor_load %b_desc[%c0, %i] : !tt.tensordesc<64x256xf16, #shared> -> tensor<64x256xf16, #blocked1>
      %b_smem = ttg.local_alloc %b : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
      %dot = ttng.warp_group_dot %a_smem, %b_smem, %iter {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
      scf.yield %dot : tensor<128x256xf32, #mma>
    } {tt.num_stages = 2 : i32, tt.warp_specialize}
    %out = arith.truncf %acc : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
    %out_blocked = ttg.convert_layout %out : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
    %independent = tt.descriptor_load %independent_desc[%c0, %c0] : !tt.tensordesc<128x256xf16, #shared> -> tensor<128x256xf16, #blocked1>
    %output_indices = arith.constant dense<0> : tensor<128x256xi32, #blocked1>
    %output_gather = tt.gather %independent[%output_indices] {axis = 1 : i32} : (tensor<128x256xf16, #blocked1>, tensor<128x256xi32, #blocked1>) -> tensor<128x256xf16, #blocked1>
    %gather_ptrs = tt.splat %arg4 : !tt.ptr<f16> -> tensor<128x256x!tt.ptr<f16>, #blocked1>
    %mask = arith.constant dense<false> : tensor<128x256xi1, #blocked1>
    tt.descriptor_store %c_desc[%c0, %c0], %out_blocked : !tt.tensordesc<128x256xf16, #shared>, tensor<128x256xf16, #blocked1>
    // Consumer partitions repeat this independent store, so keep it inert.
    tt.store %gather_ptrs, %output_gather, %mask : tensor<128x256x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}

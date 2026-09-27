// RUN: triton-opt %s --nvgpu-warp-specialization=num-stages=2 | FileCheck %s

// CHECK-LABEL: @gather_falls_back
// CHECK-NOT: ttg.warp_specialize
// CHECK: tt.gather
// CHECK: ttng.warp_group_dot
// CHECK-NOT: tt.gather
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-LABEL: @atomic_rmw_falls_back
// CHECK-NOT: ttg.warp_specialize
// CHECK: ttng.warp_group_dot
// CHECK: tt.atomic_rmw fadd
// CHECK-NOT: tt.atomic_rmw fadd
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-LABEL: @atomic_cas_falls_back
// CHECK-NOT: ttg.warp_specialize
// CHECK: ttng.warp_group_dot
// CHECK: tt.atomic_cas
// CHECK-NOT: tt.atomic_cas
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-LABEL: @producer_work_keeps_warp_specialization
// CHECK: ttg.warp_specialize
// CHECK: tt.load %arg4
// CHECK: tt.atomic_rmw add, relaxed, gpu, %arg3
// CHECK-NOT: tt.atomic_rmw add

// CHECK-LABEL: @gather_before_loop_falls_back
// CHECK-NOT: ttg.warp_specialize
// CHECK: tt.gather
// CHECK: ttng.warp_group_dot
// CHECK-NOT: tt.gather
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-LABEL: @atomic_epilogue_falls_back
// CHECK-NOT: ttg.warp_specialize
// CHECK: ttng.warp_group_dot
// CHECK: tt.atomic_rmw fadd
// CHECK-NOT: tt.atomic_rmw fadd
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
  tt.func @gather_falls_back(%arg0: !tt.tensordesc<128x64xf16>, %arg1: !tt.tensordesc<64x256xf16>, %arg2: !tt.tensordesc<128x256xf16>, %iterations: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %indices = arith.constant dense<1> : tensor<128x64xi32, #blocked>
    %init = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %acc = scf.for %i = %c0 to %iterations step %c1 iter_args(%iter = %init) -> tensor<128x256xf32, #mma> {
      %a = tt.descriptor_load %arg0[%i, %c0] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked>
      %gathered = tt.gather %a[%indices] {axis = 1 : i32} : (tensor<128x64xf16, #blocked>, tensor<128x64xi32, #blocked>) -> tensor<128x64xf16, #blocked>
      %a_smem = ttg.local_alloc %gathered : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b = tt.descriptor_load %arg1[%c0, %i] : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #blocked1>
      %b_smem = ttg.local_alloc %b : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
      %dot = ttng.warp_group_dot %a_smem, %b_smem, %iter {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
      scf.yield %dot : tensor<128x256xf32, #mma>
    } {tt.num_stages = 2 : i32, tt.warp_specialize}
    %out = arith.truncf %acc : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
    %out_blocked = ttg.convert_layout %out : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
    tt.descriptor_store %arg2[%c0, %c0], %out_blocked : !tt.tensordesc<128x256xf16>, tensor<128x256xf16, #blocked1>
    tt.return
  }

  tt.func @atomic_rmw_falls_back(%arg0: !tt.tensordesc<128x64xf16>, %arg1: !tt.tensordesc<64x256xf16>, %arg2: !tt.tensordesc<128x256xf16>, %atomic_ptr: !tt.ptr<f32>, %iterations: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %init = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %acc = scf.for %i = %c0 to %iterations step %c1 iter_args(%iter = %init) -> tensor<128x256xf32, #mma> {
      %a = tt.descriptor_load %arg0[%i, %c0] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked>
      %a_smem = ttg.local_alloc %a : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b = tt.descriptor_load %arg1[%c0, %i] : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #blocked1>
      %b_smem = ttg.local_alloc %b : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
      %dot = ttng.warp_group_dot %a_smem, %b_smem, %iter {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
      %value = ttg.convert_layout %dot : tensor<128x256xf32, #mma> -> tensor<128x256xf32, #blocked1>
      %ptrs = tt.splat %atomic_ptr : !tt.ptr<f32> -> tensor<128x256x!tt.ptr<f32>, #blocked1>
      %mask = arith.constant dense<true> : tensor<128x256xi1, #blocked1>
      %atomic = tt.atomic_rmw fadd, relaxed, gpu, %ptrs, %value, %mask : (tensor<128x256x!tt.ptr<f32>, #blocked1>, tensor<128x256xf32, #blocked1>, tensor<128x256xi1, #blocked1>) -> tensor<128x256xf32, #blocked1>
      scf.yield %dot : tensor<128x256xf32, #mma>
    } {tt.num_stages = 2 : i32, tt.warp_specialize}
    %out = arith.truncf %acc : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
    %out_blocked = ttg.convert_layout %out : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
    tt.descriptor_store %arg2[%c0, %c0], %out_blocked : !tt.tensordesc<128x256xf16>, tensor<128x256xf16, #blocked1>
    tt.return
  }

  tt.func @atomic_cas_falls_back(%arg0: !tt.tensordesc<128x64xf16>, %arg1: !tt.tensordesc<64x256xf16>, %arg2: !tt.tensordesc<128x256xf16>, %atomic_ptr: !tt.ptr<i32>, %iterations: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %init = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %acc = scf.for %i = %c0 to %iterations step %c1 iter_args(%iter = %init) -> tensor<128x256xf32, #mma> {
      %a = tt.descriptor_load %arg0[%i, %c0] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked>
      %a_smem = ttg.local_alloc %a : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b = tt.descriptor_load %arg1[%c0, %i] : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #blocked1>
      %b_smem = ttg.local_alloc %b : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
      %dot = ttng.warp_group_dot %a_smem, %b_smem, %iter {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
      %cas = tt.atomic_cas relaxed, gpu, %atomic_ptr, %c0, %c1 : (!tt.ptr<i32>, i32, i32) -> i32
      scf.yield %dot : tensor<128x256xf32, #mma>
    } {tt.num_stages = 2 : i32, tt.warp_specialize}
    %out = arith.truncf %acc : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
    %out_blocked = ttg.convert_layout %out : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
    tt.descriptor_store %arg2[%c0, %c0], %out_blocked : !tt.tensordesc<128x256xf16>, tensor<128x256xf16, #blocked1>
    tt.return
  }

  tt.func @producer_work_keeps_warp_specialization(%arg0: !tt.tensordesc<128x64xf16>, %arg1: !tt.tensordesc<64x256xf16>, %arg2: !tt.tensordesc<128x256xf16>, %arg3: !tt.ptr<i32>, %arg4: !tt.ptr<i32>, %iterations: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %init = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %acc = scf.for %i = %c0 to %iterations step %c1 iter_args(%iter = %init) -> tensor<128x256xf32, #mma> {
      %offset = tt.load %arg4 : !tt.ptr<i32>
      %coordinate = tt.atomic_rmw add, relaxed, gpu, %arg3, %c1 : (!tt.ptr<i32>, i32) -> i32
      %index = arith.addi %offset, %coordinate : i32
      %a = tt.descriptor_load %arg0[%index, %c0] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked>
      %a_smem = ttg.local_alloc %a : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b = tt.descriptor_load %arg1[%c0, %index] : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #blocked1>
      %b_smem = ttg.local_alloc %b : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
      %dot = ttng.warp_group_dot %a_smem, %b_smem, %iter {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
      scf.yield %dot : tensor<128x256xf32, #mma>
    } {tt.num_stages = 2 : i32, tt.warp_specialize}
    %out = arith.truncf %acc : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
    %out_blocked = ttg.convert_layout %out : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
    tt.descriptor_store %arg2[%c0, %c0], %out_blocked : !tt.tensordesc<128x256xf16>, tensor<128x256xf16, #blocked1>
    tt.return
  }

  tt.func @gather_before_loop_falls_back(%arg0: !tt.tensordesc<128x64xf16>, %arg1: !tt.tensordesc<64x256xf16>, %arg2: !tt.tensordesc<128x256xf16>, %iterations: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %indices = arith.constant dense<1> : tensor<128x64xi32, #blocked>
    %a = tt.descriptor_load %arg0[%c0, %c0] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked>
    %gathered = tt.gather %a[%indices] {axis = 1 : i32} : (tensor<128x64xf16, #blocked>, tensor<128x64xi32, #blocked>) -> tensor<128x64xf16, #blocked>
    %init = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %acc = scf.for %i = %c0 to %iterations step %c1 iter_args(%iter = %init) -> tensor<128x256xf32, #mma> {
      %a_smem = ttg.local_alloc %gathered : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b = tt.descriptor_load %arg1[%c0, %i] : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #blocked1>
      %b_smem = ttg.local_alloc %b : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
      %dot = ttng.warp_group_dot %a_smem, %b_smem, %iter {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
      scf.yield %dot : tensor<128x256xf32, #mma>
    } {tt.num_stages = 2 : i32, tt.warp_specialize}
    %out = arith.truncf %acc : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
    %out_blocked = ttg.convert_layout %out : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
    tt.descriptor_store %arg2[%c0, %c0], %out_blocked : !tt.tensordesc<128x256xf16>, tensor<128x256xf16, #blocked1>
    tt.return
  }

  tt.func @atomic_epilogue_falls_back(%arg0: !tt.tensordesc<128x64xf16>, %arg1: !tt.tensordesc<64x256xf16>, %arg2: !tt.tensordesc<128x256xf16>, %atomic_ptr: !tt.ptr<f32>, %iterations: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %init = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %acc = scf.for %i = %c0 to %iterations step %c1 iter_args(%iter = %init) -> tensor<128x256xf32, #mma> {
      %a = tt.descriptor_load %arg0[%i, %c0] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked>
      %a_smem = ttg.local_alloc %a : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b = tt.descriptor_load %arg1[%c0, %i] : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #blocked1>
      %b_smem = ttg.local_alloc %b : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
      %dot = ttng.warp_group_dot %a_smem, %b_smem, %iter {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
      scf.yield %dot : tensor<128x256xf32, #mma>
    } {tt.num_stages = 2 : i32, tt.warp_specialize}
    %value = ttg.convert_layout %acc : tensor<128x256xf32, #mma> -> tensor<128x256xf32, #blocked1>
    %ptrs = tt.splat %atomic_ptr : !tt.ptr<f32> -> tensor<128x256x!tt.ptr<f32>, #blocked1>
    %mask = arith.constant dense<true> : tensor<128x256xi1, #blocked1>
    %atomic = tt.atomic_rmw fadd, relaxed, gpu, %ptrs, %value, %mask : (tensor<128x256x!tt.ptr<f32>, #blocked1>, tensor<128x256xf32, #blocked1>, tensor<128x256xi1, #blocked1>) -> tensor<128x256xf32, #blocked1>
    %out = arith.truncf %acc : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
    %out_blocked = ttg.convert_layout %out : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
    tt.descriptor_store %arg2[%c0, %c0], %out_blocked : !tt.tensordesc<128x256xf16>, tensor<128x256xf16, #blocked1>
    tt.return
  }
}

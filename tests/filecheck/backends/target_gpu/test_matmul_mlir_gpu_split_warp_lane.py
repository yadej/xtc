# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: mlir-target=nvgpu

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend
from xtc.runtimes.accelerator.gpu import GPUDevice

gpu = GPUDevice()
I, J, K, dtype = 128, 128, 128, "float32"
a = O.tensor((I, K), dtype, name="A", device=gpu)
b = O.tensor((K, J), dtype, name="B", device=gpu)

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C", device=gpu)

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler()
sch.set_dims(["i", "j", "k"])
sch.tile("i", {"i1": 4, "i2": 4})
sch.tile("j", {"j1": 64, "j2": 4, "j3": 4})
sch.tile("k", {"k2": 16})
sch.split("k", {"k_lo": 0, "k_hi": 64})
sch.gpu_block(["j", "i"])
sch.gpu_warp(["j1"])
sch.gpu_lane(["j2", "i1"])
sch.interchange(["j", "i", "j1", "j2", "i1", "k_lo", "k_hi"])
# Both split sections are scheduled below the gpu-mapped loops, each
# one with its own loop order
sch.interchange(["k", "j3", "i2", "k2"], root="./k_lo")
sch.interchange(["k", "j3", "i2", "k2"], root="./k_hi")
sched = sch.schedule()

comp = impl.get_compiler(
    target=gpu,
    shared_lib=True,
    dump_file="matmul_gpu_split_warp_lane",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)

evaluator = module.get_evaluator(validate=True)
results, code, error = evaluator.evaluate()
print(f"CODE: {code}")

# CHECK:       // -----// IR Dump Before transform //----- //
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: memref<128x128xf32> {llvm.noalias, memref.on_device}, %arg1: memref<128x128xf32> {llvm.noalias, memref.on_device}, %arg2: memref<128x128xf32> {llvm.noalias, memref.on_device}) {
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%arg2 : memref<128x128xf32>)
# CHECK-NEXT:      linalg.matmul {__xtc_id_C_} ins(%arg0, %arg1 : memref<128x128xf32>, memref<128x128xf32>) outs(%arg2 : memref<128x128xf32>)
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:      transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = transform.structured.match attributes {__xtc_id_C_0_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops "./i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_1 "./j" : !transform.any_op
# CHECK-NEXT:      %1 = transform.structured.match attributes {__xtc_id_C_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_op, %forall_op = transform.structured.tile_using_forall %1 tile_sizes [4, 64, 0](mapping = [#gpu.block<x>, #gpu.block<y>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %forall_op "./j" : !transform.any_op
# CHECK-NEXT:      %tiled_op_2, %forall_op_3 = transform.structured.tile_using_forall %tiled_op tile_sizes [0, 4, 0](mapping = [#gpu.warp<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %forall_op_3 "./j1" : !transform.any_op
# CHECK-NEXT:      %tiled_op_4, %forall_op_5 = transform.structured.tile_using_forall %tiled_op_2 tile_sizes [4, 4, 0](mapping = [#gpu.lane<linear_dim_0>, #gpu.lane<linear_dim_1>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %forall_op_5 "./j2" : !transform.any_op
# CHECK-NEXT:      %2 = transform.structured.split %tiled_op_4 after 64  {dimension = 2 : i64} : !transform.any_op
# CHECK-NEXT:      %3:2 = transform.split_handle %2 {fail_on_payload_too_small = false} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %3#0 tile_sizes [0, 0, 16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_7 "./k_lo/k" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_9 "./k_lo/j3" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_10, %loops_11 = transform.structured.tile_using_for %tiled_linalg_op_8 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_11 "./k_lo/i2" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_12, %loops_13 = transform.structured.tile_using_for %tiled_linalg_op_10 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_13 "./k_lo/k2" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_14, %loops_15 = transform.structured.tile_using_for %3#1 tile_sizes [0, 0, 16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_15 "./k_hi/k" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_16, %loops_17 = transform.structured.tile_using_for %tiled_linalg_op_14 tile_sizes [0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_17 "./k_hi/j3" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_18, %loops_19 = transform.structured.tile_using_for %tiled_linalg_op_16 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_19 "./k_hi/i2" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_20, %loops_21 = transform.structured.tile_using_for %tiled_linalg_op_18 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_21 "./k_hi/k2" : !transform.any_op
# CHECK-NEXT:      %4 = transform.gpu.map_forall_to_blocks %forall_op generate_gpu_launch : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %5 = transform.gpu.map_nested_forall_to_threads %4 block_dims = [512, 1, 1] : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0) -> (d0 * 4)>
# CHECK-NEXT:  #map1 = affine_map<(d0) -> (d0 * 64)>
# CHECK-NEXT:  #map2 = affine_map<()[s0] -> (s0 floordiv 32)>
# CHECK-NEXT:  #map3 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 512 + s2 * 512)>
# CHECK-NEXT:  #map4 = affine_map<()[s0] -> (s0 mod 32)>
# CHECK-NEXT:  #map5 = affine_map<() -> (0)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: memref<128x128xf32> {llvm.noalias, memref.on_device}, %arg1: memref<128x128xf32> {llvm.noalias, memref.on_device}, %arg2: memref<128x128xf32> {llvm.noalias, memref.on_device}) {
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c128 = arith.constant 128 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c128 step %c1 {
# CHECK-NEXT:        %subview = memref.subview %arg2[%arg3, 0] [1, 128] [1, 1] : memref<128x128xf32> to memref<1x128xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %c0_4 = arith.constant 0 : index
# CHECK-NEXT:        %c128_5 = arith.constant 128 : index
# CHECK-NEXT:        %c1_6 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg4 = %c0_4 to %c128_5 step %c1_6 {
# CHECK-NEXT:          %subview_7 = memref.subview %subview[0, %arg4] [1, 1] [1, 1] : memref<1x128xf32, strided<[128, 1], offset: ?>> to memref<1x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:          linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%subview_7 : memref<1x1xf32, strided<[128, 1], offset: ?>>)
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %c1_0 = arith.constant 1 : index
# CHECK-NEXT:      %c512 = arith.constant 512 : index
# CHECK-NEXT:      %c1_1 = arith.constant 1 : index
# CHECK-NEXT:      %c1_2 = arith.constant 1 : index
# CHECK-NEXT:      %c32 = arith.constant 32 : index
# CHECK-NEXT:      %c2 = arith.constant 2 : index
# CHECK-NEXT:      %c1_3 = arith.constant 1 : index
# CHECK-NEXT:      gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c32, %arg10 = %c2, %arg11 = %c1_3) threads(%arg6, %arg7, %arg8) in (%arg12 = %c512, %arg13 = %c1_1, %arg14 = %c1_2) {
# CHECK-NEXT:        %c0_4 = arith.constant 0 : index
# CHECK-NEXT:        %c0_5 = arith.constant 0 : index
# CHECK-NEXT:        %block_id_x = gpu.block_id  x
# CHECK-NEXT:        %block_id_y = gpu.block_id  y
# CHECK-NEXT:        %block_id_z = gpu.block_id  z
# CHECK-NEXT:        %0 = affine.apply #map(%block_id_x)
# CHECK-NEXT:        %1 = affine.apply #map1(%block_id_y)
# CHECK-NEXT:        %subview = memref.subview %arg0[%0, 0] [4, 128] [1, 1] : memref<128x128xf32> to memref<4x128xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %subview_6 = memref.subview %arg1[0, %1] [128, 64] [1, 1] : memref<128x128xf32> to memref<128x64xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %subview_7 = memref.subview %arg2[%0, %1] [4, 64] [1, 1] : memref<128x128xf32> to memref<4x64xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %thread_id_x = gpu.thread_id  x
# CHECK-NEXT:        %thread_id_y = gpu.thread_id  y
# CHECK-NEXT:        %thread_id_z = gpu.thread_id  z
# CHECK-NEXT:        %2 = affine.apply #map2()[%thread_id_x]
# CHECK-NEXT:        %3 = affine.apply #map(%2)
# CHECK-NEXT:        %subview_8 = memref.subview %subview[0, 0] [4, 128] [1, 1] : memref<4x128xf32, strided<[128, 1], offset: ?>> to memref<4x128xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %subview_9 = memref.subview %subview_6[0, %3] [128, 4] [1, 1] : memref<128x64xf32, strided<[128, 1], offset: ?>> to memref<128x4xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %subview_10 = memref.subview %subview_7[0, %3] [4, 4] [1, 1] : memref<4x64xf32, strided<[128, 1], offset: ?>> to memref<4x4xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %thread_id_x_11 = gpu.thread_id  x
# CHECK-NEXT:        %thread_id_y_12 = gpu.thread_id  y
# CHECK-NEXT:        %thread_id_z_13 = gpu.thread_id  z
# CHECK-NEXT:        %4 = affine.apply #map3()[%thread_id_x_11, %c0_4, %c0_4]
# CHECK-NEXT:        %5 = affine.apply #map4()[%thread_id_x_11]
# CHECK-NEXT:        %6 = affine.apply #map5()
# CHECK-NEXT:        %7 = affine.apply #map4()[%thread_id_x_11]
# CHECK-NEXT:        %c1_14 = arith.constant 1 : index
# CHECK-NEXT:        %8 = arith.cmpi ult, %5, %c1_14 : index
# CHECK-NEXT:        scf.if %8 {
# CHECK-NEXT:          %9 = affine.apply #map(%6)
# CHECK-NEXT:          %10 = affine.apply #map(%7)
# CHECK-NEXT:          %subview_15 = memref.subview %subview_8[%9, 0] [4, 128] [1, 1] : memref<4x128xf32, strided<[128, 1], offset: ?>> to memref<4x128xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:          %subview_16 = memref.subview %subview_9[0, %10] [128, 4] [1, 1] : memref<128x4xf32, strided<[128, 1], offset: ?>> to memref<128x4xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:          %subview_17 = memref.subview %subview_10[%9, %10] [4, 4] [1, 1] : memref<4x4xf32, strided<[128, 1], offset: ?>> to memref<4x4xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:          %subview_18 = memref.subview %subview_15[0, 0] [4, 64] [1, 1] : memref<4x128xf32, strided<[128, 1], offset: ?>> to memref<4x64xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:          %subview_19 = memref.subview %subview_16[0, 0] [64, 4] [1, 1] : memref<128x4xf32, strided<[128, 1], offset: ?>> to memref<64x4xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:          %subview_20 = memref.subview %subview_17[0, 0] [4, 4] [1, 1] : memref<4x4xf32, strided<[128, 1], offset: ?>> to memref<4x4xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:          %c0_21 = arith.constant 0 : index
# CHECK-NEXT:          %c64 = arith.constant 64 : index
# CHECK-NEXT:          %c16 = arith.constant 16 : index
# CHECK-NEXT:          scf.for %arg15 = %c0_21 to %c64 step %c16 {
# CHECK-NEXT:            %subview_28 = memref.subview %subview_18[0, %arg15] [4, 16] [1, 1] : memref<4x64xf32, strided<[128, 1], offset: ?>> to memref<4x16xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:            %subview_29 = memref.subview %subview_19[%arg15, 0] [16, 4] [1, 1] : memref<64x4xf32, strided<[128, 1], offset: ?>> to memref<16x4xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:            %subview_30 = memref.subview %subview_20[0, 0] [4, 4] [1, 1] : memref<4x4xf32, strided<[128, 1], offset: ?>> to memref<4x4xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:            %c0_31 = arith.constant 0 : index
# CHECK-NEXT:            %c4 = arith.constant 4 : index
# CHECK-NEXT:            %c1_32 = arith.constant 1 : index
# CHECK-NEXT:            scf.for %arg16 = %c0_31 to %c4 step %c1_32 {
# CHECK-NEXT:              %subview_33 = memref.subview %subview_28[0, 0] [4, 16] [1, 1] : memref<4x16xf32, strided<[128, 1], offset: ?>> to memref<4x16xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:              %subview_34 = memref.subview %subview_29[0, %arg16] [16, 1] [1, 1] : memref<16x4xf32, strided<[128, 1], offset: ?>> to memref<16x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:              %subview_35 = memref.subview %subview_30[0, %arg16] [4, 1] [1, 1] : memref<4x4xf32, strided<[128, 1], offset: ?>> to memref<4x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:              %c0_36 = arith.constant 0 : index
# CHECK-NEXT:              %c4_37 = arith.constant 4 : index
# CHECK-NEXT:              %c1_38 = arith.constant 1 : index
# CHECK-NEXT:              scf.for %arg17 = %c0_36 to %c4_37 step %c1_38 {
# CHECK-NEXT:                %subview_39 = memref.subview %subview_33[%arg17, 0] [1, 16] [1, 1] : memref<4x16xf32, strided<[128, 1], offset: ?>> to memref<1x16xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:                %subview_40 = memref.subview %subview_34[0, 0] [16, 1] [1, 1] : memref<16x1xf32, strided<[128, 1], offset: ?>> to memref<16x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:                %subview_41 = memref.subview %subview_35[%arg17, 0] [1, 1] [1, 1] : memref<4x1xf32, strided<[128, 1], offset: ?>> to memref<1x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:                %c0_42 = arith.constant 0 : index
# CHECK-NEXT:                %c16_43 = arith.constant 16 : index
# CHECK-NEXT:                %c1_44 = arith.constant 1 : index
# CHECK-NEXT:                scf.for %arg18 = %c0_42 to %c16_43 step %c1_44 {
# CHECK-NEXT:                  %subview_45 = memref.subview %subview_39[0, %arg18] [1, 1] [1, 1] : memref<1x16xf32, strided<[128, 1], offset: ?>> to memref<1x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:                  %subview_46 = memref.subview %subview_40[%arg18, 0] [1, 1] [1, 1] : memref<16x1xf32, strided<[128, 1], offset: ?>> to memref<1x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:                  %subview_47 = memref.subview %subview_41[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[128, 1], offset: ?>> to memref<1x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:                  linalg.matmul {__xtc_id_C_} ins(%subview_45, %subview_46 : memref<1x1xf32, strided<[128, 1], offset: ?>>, memref<1x1xf32, strided<[128, 1], offset: ?>>) outs(%subview_47 : memref<1x1xf32, strided<[128, 1], offset: ?>>)
# CHECK-NEXT:                } {"./k_lo/k2"}
# CHECK-NEXT:              } {"./k_lo/i2"}
# CHECK-NEXT:            } {"./k_lo/j3"}
# CHECK-NEXT:          } {"./k_lo/k"}
# CHECK-NEXT:          %subview_22 = memref.subview %subview_15[0, 64] [4, 64] [1, 1] : memref<4x128xf32, strided<[128, 1], offset: ?>> to memref<4x64xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:          %subview_23 = memref.subview %subview_16[64, 0] [64, 4] [1, 1] : memref<128x4xf32, strided<[128, 1], offset: ?>> to memref<64x4xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:          %subview_24 = memref.subview %subview_17[0, 0] [4, 4] [1, 1] : memref<4x4xf32, strided<[128, 1], offset: ?>> to memref<4x4xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:          %c0_25 = arith.constant 0 : index
# CHECK-NEXT:          %c64_26 = arith.constant 64 : index
# CHECK-NEXT:          %c16_27 = arith.constant 16 : index
# CHECK-NEXT:          scf.for %arg15 = %c0_25 to %c64_26 step %c16_27 {
# CHECK-NEXT:            %subview_28 = memref.subview %subview_22[0, %arg15] [4, 16] [1, 1] : memref<4x64xf32, strided<[128, 1], offset: ?>> to memref<4x16xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:            %subview_29 = memref.subview %subview_23[%arg15, 0] [16, 4] [1, 1] : memref<64x4xf32, strided<[128, 1], offset: ?>> to memref<16x4xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:            %subview_30 = memref.subview %subview_24[0, 0] [4, 4] [1, 1] : memref<4x4xf32, strided<[128, 1], offset: ?>> to memref<4x4xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:            %c0_31 = arith.constant 0 : index
# CHECK-NEXT:            %c4 = arith.constant 4 : index
# CHECK-NEXT:            %c1_32 = arith.constant 1 : index
# CHECK-NEXT:            scf.for %arg16 = %c0_31 to %c4 step %c1_32 {
# CHECK-NEXT:              %subview_33 = memref.subview %subview_28[0, 0] [4, 16] [1, 1] : memref<4x16xf32, strided<[128, 1], offset: ?>> to memref<4x16xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:              %subview_34 = memref.subview %subview_29[0, %arg16] [16, 1] [1, 1] : memref<16x4xf32, strided<[128, 1], offset: ?>> to memref<16x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:              %subview_35 = memref.subview %subview_30[0, %arg16] [4, 1] [1, 1] : memref<4x4xf32, strided<[128, 1], offset: ?>> to memref<4x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:              %c0_36 = arith.constant 0 : index
# CHECK-NEXT:              %c4_37 = arith.constant 4 : index
# CHECK-NEXT:              %c1_38 = arith.constant 1 : index
# CHECK-NEXT:              scf.for %arg17 = %c0_36 to %c4_37 step %c1_38 {
# CHECK-NEXT:                %subview_39 = memref.subview %subview_33[%arg17, 0] [1, 16] [1, 1] : memref<4x16xf32, strided<[128, 1], offset: ?>> to memref<1x16xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:                %subview_40 = memref.subview %subview_34[0, 0] [16, 1] [1, 1] : memref<16x1xf32, strided<[128, 1], offset: ?>> to memref<16x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:                %subview_41 = memref.subview %subview_35[%arg17, 0] [1, 1] [1, 1] : memref<4x1xf32, strided<[128, 1], offset: ?>> to memref<1x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:                %c0_42 = arith.constant 0 : index
# CHECK-NEXT:                %c16_43 = arith.constant 16 : index
# CHECK-NEXT:                %c1_44 = arith.constant 1 : index
# CHECK-NEXT:                scf.for %arg18 = %c0_42 to %c16_43 step %c1_44 {
# CHECK-NEXT:                  %subview_45 = memref.subview %subview_39[0, %arg18] [1, 1] [1, 1] : memref<1x16xf32, strided<[128, 1], offset: ?>> to memref<1x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:                  %subview_46 = memref.subview %subview_40[%arg18, 0] [1, 1] [1, 1] : memref<16x1xf32, strided<[128, 1], offset: ?>> to memref<1x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:                  %subview_47 = memref.subview %subview_41[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[128, 1], offset: ?>> to memref<1x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:                  linalg.matmul {__xtc_id_C_} ins(%subview_45, %subview_46 : memref<1x1xf32, strided<[128, 1], offset: ?>>, memref<1x1xf32, strided<[128, 1], offset: ?>>) outs(%subview_47 : memref<1x1xf32, strided<[128, 1], offset: ?>>)
# CHECK-NEXT:                } {"./k_hi/k2"}
# CHECK-NEXT:              } {"./k_hi/i2"}
# CHECK-NEXT:            } {"./k_hi/j3"}
# CHECK-NEXT:          } {"./k_hi/k"}
# CHECK-NEXT:        }
# CHECK-NEXT:        gpu.barrier
# CHECK-NEXT:        gpu.barrier
# CHECK-NEXT:        gpu.terminator
# CHECK-NEXT:      }
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  graph:
# CHECK-NEXT:    name: matmul
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 128x128xfloat32
# CHECK-NEXT:    - %1 : 128x128xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %2 : 128x128xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: matmul(%0, %1) {name = 'C'} : [128x128xfloat32, 128x128xfloat32] -> [128x128xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  CODE: 0

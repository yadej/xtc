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
sch.tile("i", {"i1": 16, "i2": 4})
sch.tile("j", {"j1": 16, "j2": 4})
sch.split("k", {"k_lo": 0, "k_hi": 64})
sch.gpu_block(["i", "j"])
sch.gpu_thread(["i1", "j1"])
sch.interchange(["i", "j", "i1", "j1", "k_lo", "k_hi"])
sch.interchange(["k", "i2", "j2"], root="./k_lo")
sch.interchange(["k", "i2", "j2"], root="./k_hi")
sched = sch.schedule()

comp = impl.get_compiler(
    target=gpu,
    shared_lib=True,
    dump_file="matmul_gpu_split",
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
# CHECK-NEXT:      %tiled_op, %forall_op = transform.structured.tile_using_forall %1 tile_sizes [16, 16, 0](mapping = [#gpu.block<x>, #gpu.block<y>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %forall_op "./i" : !transform.any_op
# CHECK-NEXT:      %tiled_op_2, %forall_op_3 = transform.structured.tile_using_forall %tiled_op tile_sizes [4, 4, 0](mapping = [#gpu.thread<x>, #gpu.thread<y>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %forall_op_3 "./i1" : !transform.any_op
# CHECK-NEXT:      %2 = transform.structured.split %tiled_op_2 after 64  {dimension = 2 : i64} : !transform.any_op
# CHECK-NEXT:      %3:2 = transform.split_handle %2 {fail_on_payload_too_small = false} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %3#0 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_5 "./k_lo/k" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %tiled_linalg_op_4 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_7 "./k_lo/i2" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_9 "./k_lo/j2" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_10, %loops_11 = transform.structured.tile_using_for %3#1 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_11 "./k_hi/k" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_12, %loops_13 = transform.structured.tile_using_for %tiled_linalg_op_10 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_13 "./k_hi/i2" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_14, %loops_15 = transform.structured.tile_using_for %tiled_linalg_op_12 tile_sizes [0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_15 "./k_hi/j2" : !transform.any_op
# CHECK-NEXT:      %4 = transform.gpu.map_forall_to_blocks %forall_op generate_gpu_launch : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %5 = transform.gpu.map_nested_forall_to_threads %4 block_dims = [4, 4, 1] : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0) -> (d0 * 16)>
# CHECK-NEXT:  #map1 = affine_map<(d0) -> (d0 * 4)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: memref<128x128xf32> {llvm.noalias, memref.on_device}, %arg1: memref<128x128xf32> {llvm.noalias, memref.on_device}, %arg2: memref<128x128xf32> {llvm.noalias, memref.on_device}) {
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c128 = arith.constant 128 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c128 step %c1 {
# CHECK-NEXT:        %subview = memref.subview %arg2[%arg3, 0] [1, 128] [1, 1] : memref<128x128xf32> to memref<1x128xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %c0_5 = arith.constant 0 : index
# CHECK-NEXT:        %c128_6 = arith.constant 128 : index
# CHECK-NEXT:        %c1_7 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg4 = %c0_5 to %c128_6 step %c1_7 {
# CHECK-NEXT:          %subview_8 = memref.subview %subview[0, %arg4] [1, 1] [1, 1] : memref<1x128xf32, strided<[128, 1], offset: ?>> to memref<1x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:          linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%subview_8 : memref<1x1xf32, strided<[128, 1], offset: ?>>)
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %c1_0 = arith.constant 1 : index
# CHECK-NEXT:      %c4 = arith.constant 4 : index
# CHECK-NEXT:      %c4_1 = arith.constant 4 : index
# CHECK-NEXT:      %c1_2 = arith.constant 1 : index
# CHECK-NEXT:      %c8 = arith.constant 8 : index
# CHECK-NEXT:      %c8_3 = arith.constant 8 : index
# CHECK-NEXT:      %c1_4 = arith.constant 1 : index
# CHECK-NEXT:      gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c8, %arg10 = %c8_3, %arg11 = %c1_4) threads(%arg6, %arg7, %arg8) in (%arg12 = %c4, %arg13 = %c4_1, %arg14 = %c1_2) {
# CHECK-NEXT:        %c0_5 = arith.constant 0 : index
# CHECK-NEXT:        %c0_6 = arith.constant 0 : index
# CHECK-NEXT:        %block_id_x = gpu.block_id  x
# CHECK-NEXT:        %block_id_y = gpu.block_id  y
# CHECK-NEXT:        %block_id_z = gpu.block_id  z
# CHECK-NEXT:        %0 = affine.apply #map(%block_id_x)
# CHECK-NEXT:        %1 = affine.apply #map(%block_id_y)
# CHECK-NEXT:        %subview = memref.subview %arg0[%0, 0] [16, 128] [1, 1] : memref<128x128xf32> to memref<16x128xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %subview_7 = memref.subview %arg1[0, %1] [128, 16] [1, 1] : memref<128x128xf32> to memref<128x16xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %subview_8 = memref.subview %arg2[%0, %1] [16, 16] [1, 1] : memref<128x128xf32> to memref<16x16xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %thread_id_x = gpu.thread_id  x
# CHECK-NEXT:        %thread_id_y = gpu.thread_id  y
# CHECK-NEXT:        %thread_id_z = gpu.thread_id  z
# CHECK-NEXT:        %2 = affine.apply #map1(%thread_id_x)
# CHECK-NEXT:        %3 = affine.apply #map1(%thread_id_y)
# CHECK-NEXT:        %subview_9 = memref.subview %subview[%2, 0] [4, 128] [1, 1] : memref<16x128xf32, strided<[128, 1], offset: ?>> to memref<4x128xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %subview_10 = memref.subview %subview_7[0, %3] [128, 4] [1, 1] : memref<128x16xf32, strided<[128, 1], offset: ?>> to memref<128x4xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %subview_11 = memref.subview %subview_8[%2, %3] [4, 4] [1, 1] : memref<16x16xf32, strided<[128, 1], offset: ?>> to memref<4x4xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %subview_12 = memref.subview %subview_9[0, 0] [4, 64] [1, 1] : memref<4x128xf32, strided<[128, 1], offset: ?>> to memref<4x64xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %subview_13 = memref.subview %subview_10[0, 0] [64, 4] [1, 1] : memref<128x4xf32, strided<[128, 1], offset: ?>> to memref<64x4xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %subview_14 = memref.subview %subview_11[0, 0] [4, 4] [1, 1] : memref<4x4xf32, strided<[128, 1], offset: ?>> to memref<4x4xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %c0_15 = arith.constant 0 : index
# CHECK-NEXT:        %c64 = arith.constant 64 : index
# CHECK-NEXT:        %c1_16 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg15 = %c0_15 to %c64 step %c1_16 {
# CHECK-NEXT:          %subview_23 = memref.subview %subview_12[0, %arg15] [4, 1] [1, 1] : memref<4x64xf32, strided<[128, 1], offset: ?>> to memref<4x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:          %subview_24 = memref.subview %subview_13[%arg15, 0] [1, 4] [1, 1] : memref<64x4xf32, strided<[128, 1], offset: ?>> to memref<1x4xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:          %subview_25 = memref.subview %subview_14[0, 0] [4, 4] [1, 1] : memref<4x4xf32, strided<[128, 1], offset: ?>> to memref<4x4xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:          %c0_26 = arith.constant 0 : index
# CHECK-NEXT:          %c4_27 = arith.constant 4 : index
# CHECK-NEXT:          %c1_28 = arith.constant 1 : index
# CHECK-NEXT:          scf.for %arg16 = %c0_26 to %c4_27 step %c1_28 {
# CHECK-NEXT:            %subview_29 = memref.subview %subview_23[%arg16, 0] [1, 1] [1, 1] : memref<4x1xf32, strided<[128, 1], offset: ?>> to memref<1x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:            %subview_30 = memref.subview %subview_24[0, 0] [1, 4] [1, 1] : memref<1x4xf32, strided<[128, 1], offset: ?>> to memref<1x4xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:            %subview_31 = memref.subview %subview_25[%arg16, 0] [1, 4] [1, 1] : memref<4x4xf32, strided<[128, 1], offset: ?>> to memref<1x4xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:            %c0_32 = arith.constant 0 : index
# CHECK-NEXT:            %c4_33 = arith.constant 4 : index
# CHECK-NEXT:            %c1_34 = arith.constant 1 : index
# CHECK-NEXT:            scf.for %arg17 = %c0_32 to %c4_33 step %c1_34 {
# CHECK-NEXT:              %subview_35 = memref.subview %subview_29[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[128, 1], offset: ?>> to memref<1x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:              %subview_36 = memref.subview %subview_30[0, %arg17] [1, 1] [1, 1] : memref<1x4xf32, strided<[128, 1], offset: ?>> to memref<1x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:              %subview_37 = memref.subview %subview_31[0, %arg17] [1, 1] [1, 1] : memref<1x4xf32, strided<[128, 1], offset: ?>> to memref<1x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:              linalg.matmul {__xtc_id_C_} ins(%subview_35, %subview_36 : memref<1x1xf32, strided<[128, 1], offset: ?>>, memref<1x1xf32, strided<[128, 1], offset: ?>>) outs(%subview_37 : memref<1x1xf32, strided<[128, 1], offset: ?>>)
# CHECK-NEXT:            } {"./k_lo/j2"}
# CHECK-NEXT:          } {"./k_lo/i2"}
# CHECK-NEXT:        } {"./k_lo/k"}
# CHECK-NEXT:        %subview_17 = memref.subview %subview_9[0, 64] [4, 64] [1, 1] : memref<4x128xf32, strided<[128, 1], offset: ?>> to memref<4x64xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %subview_18 = memref.subview %subview_10[64, 0] [64, 4] [1, 1] : memref<128x4xf32, strided<[128, 1], offset: ?>> to memref<64x4xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %subview_19 = memref.subview %subview_11[0, 0] [4, 4] [1, 1] : memref<4x4xf32, strided<[128, 1], offset: ?>> to memref<4x4xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %c0_20 = arith.constant 0 : index
# CHECK-NEXT:        %c64_21 = arith.constant 64 : index
# CHECK-NEXT:        %c1_22 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg15 = %c0_20 to %c64_21 step %c1_22 {
# CHECK-NEXT:          %subview_23 = memref.subview %subview_17[0, %arg15] [4, 1] [1, 1] : memref<4x64xf32, strided<[128, 1], offset: ?>> to memref<4x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:          %subview_24 = memref.subview %subview_18[%arg15, 0] [1, 4] [1, 1] : memref<64x4xf32, strided<[128, 1], offset: ?>> to memref<1x4xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:          %subview_25 = memref.subview %subview_19[0, 0] [4, 4] [1, 1] : memref<4x4xf32, strided<[128, 1], offset: ?>> to memref<4x4xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:          %c0_26 = arith.constant 0 : index
# CHECK-NEXT:          %c4_27 = arith.constant 4 : index
# CHECK-NEXT:          %c1_28 = arith.constant 1 : index
# CHECK-NEXT:          scf.for %arg16 = %c0_26 to %c4_27 step %c1_28 {
# CHECK-NEXT:            %subview_29 = memref.subview %subview_23[%arg16, 0] [1, 1] [1, 1] : memref<4x1xf32, strided<[128, 1], offset: ?>> to memref<1x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:            %subview_30 = memref.subview %subview_24[0, 0] [1, 4] [1, 1] : memref<1x4xf32, strided<[128, 1], offset: ?>> to memref<1x4xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:            %subview_31 = memref.subview %subview_25[%arg16, 0] [1, 4] [1, 1] : memref<4x4xf32, strided<[128, 1], offset: ?>> to memref<1x4xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:            %c0_32 = arith.constant 0 : index
# CHECK-NEXT:            %c4_33 = arith.constant 4 : index
# CHECK-NEXT:            %c1_34 = arith.constant 1 : index
# CHECK-NEXT:            scf.for %arg17 = %c0_32 to %c4_33 step %c1_34 {
# CHECK-NEXT:              %subview_35 = memref.subview %subview_29[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[128, 1], offset: ?>> to memref<1x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:              %subview_36 = memref.subview %subview_30[0, %arg17] [1, 1] [1, 1] : memref<1x4xf32, strided<[128, 1], offset: ?>> to memref<1x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:              %subview_37 = memref.subview %subview_31[0, %arg17] [1, 1] [1, 1] : memref<1x4xf32, strided<[128, 1], offset: ?>> to memref<1x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:              linalg.matmul {__xtc_id_C_} ins(%subview_35, %subview_36 : memref<1x1xf32, strided<[128, 1], offset: ?>>, memref<1x1xf32, strided<[128, 1], offset: ?>>) outs(%subview_37 : memref<1x1xf32, strided<[128, 1], offset: ?>>)
# CHECK-NEXT:            } {"./k_hi/j2"}
# CHECK-NEXT:          } {"./k_hi/i2"}
# CHECK-NEXT:        } {"./k_hi/k"}
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

# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: mlir-target=nvgpu

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend
from xtc.schedules.descript import descript_scheduler

from xtc.runtimes.accelerator.gpu import GPUDevice

gpu = GPUDevice()
I, J, K, dtype = 512, 256, 128, "float32"
a = O.tensor((I, K), dtype, name="A", device=gpu)
b = O.tensor((K, J), dtype, name="B", device=gpu)

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C", device=gpu)

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler()
descript_scheduler(
    scheduler = sch,
    node_name = "C",
    abstract_dims = ["I","J","K"],
    spec = {
        "I[:128]": {
            "I": {"gpu_block": 0},
            "I#128": {"gpu_thread": 0},
            "I#8": {},
            "K": {},
            "J": {},
        },
        "I[128:]": {
            "I": {"gpu_block": 0},
            "I#64": {"gpu_thread": 0},
            "I#8": {},
            "K": {},
            "J": {},
        }
    }
)

sched = sch.schedule()

comp = impl.get_compiler(
    target=gpu,
    shared_lib=True,
    dump_file="matmul_descript_gpu_split",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

# CHECK:       // -----// IR Dump Before transform //----- //
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: memref<512x128xf32> {llvm.noalias, memref.on_device}, %arg1: memref<128x256xf32> {llvm.noalias, memref.on_device}, %arg2: memref<512x256xf32> {llvm.noalias, memref.on_device}) {
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%arg2 : memref<512x256xf32>)
# CHECK-NEXT:      linalg.matmul {__xtc_id_C_} ins(%arg0, %arg1 : memref<512x128xf32>, memref<128x256xf32>) outs(%arg2 : memref<512x256xf32>)
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
# CHECK-NEXT:      %2 = transform.structured.split %1 after 128  {dimension = 0 : i64} : !transform.any_op
# CHECK-NEXT:      %3:2 = transform.split_handle %2 {fail_on_payload_too_small = false} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      %tiled_op, %forall_op = transform.structured.tile_using_forall %3#0 tile_sizes [128, 0, 0](mapping = [#gpu.block<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %forall_op "C/I[0]/I" : !transform.any_op
# CHECK-NEXT:      %tiled_op_2, %forall_op_3 = transform.structured.tile_using_forall %tiled_op tile_sizes [8, 0, 0](mapping = [#gpu.thread<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %forall_op_3 "C/I[0]/I0" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_op_2 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_5 "C/I[0]/I1" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %tiled_linalg_op_4 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_7 "C/I[0]/K" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_9 "C/I[0]/J" : !transform.any_op
# CHECK-NEXT:      %tiled_op_10, %forall_op_11 = transform.structured.tile_using_forall %3#1 tile_sizes [64, 0, 0](mapping = [#gpu.block<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %forall_op_11 "C/I[1]/I" : !transform.any_op
# CHECK-NEXT:      %tiled_op_12, %forall_op_13 = transform.structured.tile_using_forall %tiled_op_10 tile_sizes [8, 0, 0](mapping = [#gpu.thread<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %forall_op_13 "C/I[1]/I0" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_14, %loops_15 = transform.structured.tile_using_for %tiled_op_12 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_15 "C/I[1]/I1" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_16, %loops_17 = transform.structured.tile_using_for %tiled_linalg_op_14 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_17 "C/I[1]/K" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_18, %loops_19 = transform.structured.tile_using_for %tiled_linalg_op_16 tile_sizes [0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_19 "C/I[1]/J" : !transform.any_op
# CHECK-NEXT:      %4 = transform.gpu.map_forall_to_blocks %forall_op generate_gpu_launch : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %5 = transform.gpu.map_nested_forall_to_threads %4 block_dims = [16, 1, 1] : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %6 = transform.gpu.map_forall_to_blocks %forall_op_11 generate_gpu_launch : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %7 = transform.gpu.map_nested_forall_to_threads %6 block_dims = [8, 1, 1] : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0) -> (d0 * 128)>
# CHECK-NEXT:  #map1 = affine_map<(d0) -> (d0 * 8)>
# CHECK-NEXT:  #map2 = affine_map<(d0) -> (d0 * 64)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: memref<512x128xf32> {llvm.noalias, memref.on_device}, %arg1: memref<128x256xf32> {llvm.noalias, memref.on_device}, %arg2: memref<512x256xf32> {llvm.noalias, memref.on_device}) {
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c512 = arith.constant 512 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c512 step %c1 {
# CHECK-NEXT:        %subview_16 = memref.subview %arg2[%arg3, 0] [1, 256] [1, 1] : memref<512x256xf32> to memref<1x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        %c0_17 = arith.constant 0 : index
# CHECK-NEXT:        %c256 = arith.constant 256 : index
# CHECK-NEXT:        %c1_18 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg4 = %c0_17 to %c256 step %c1_18 {
# CHECK-NEXT:          %subview_19 = memref.subview %subview_16[0, %arg4] [1, 1] [1, 1] : memref<1x256xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:          linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%subview_19 : memref<1x1xf32, strided<[256, 1], offset: ?>>)
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %subview = memref.subview %arg0[0, 0] [128, 128] [1, 1] : memref<512x128xf32> to memref<128x128xf32, strided<[128, 1]>>
# CHECK-NEXT:      %subview_0 = memref.subview %arg1[0, 0] [128, 256] [1, 1] : memref<128x256xf32> to memref<128x256xf32, strided<[256, 1]>>
# CHECK-NEXT:      %subview_1 = memref.subview %arg2[0, 0] [128, 256] [1, 1] : memref<512x256xf32> to memref<128x256xf32, strided<[256, 1]>>
# CHECK-NEXT:      %c1_2 = arith.constant 1 : index
# CHECK-NEXT:      %c16 = arith.constant 16 : index
# CHECK-NEXT:      %c1_3 = arith.constant 1 : index
# CHECK-NEXT:      %c1_4 = arith.constant 1 : index
# CHECK-NEXT:      %c1_5 = arith.constant 1 : index
# CHECK-NEXT:      %c1_6 = arith.constant 1 : index
# CHECK-NEXT:      %c1_7 = arith.constant 1 : index
# CHECK-NEXT:      gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c1_5, %arg10 = %c1_6, %arg11 = %c1_7) threads(%arg6, %arg7, %arg8) in (%arg12 = %c16, %arg13 = %c1_3, %arg14 = %c1_4) {
# CHECK-NEXT:        %c0_16 = arith.constant 0 : index
# CHECK-NEXT:        %c0_17 = arith.constant 0 : index
# CHECK-NEXT:        %block_id_x = gpu.block_id  x
# CHECK-NEXT:        %block_id_y = gpu.block_id  y
# CHECK-NEXT:        %block_id_z = gpu.block_id  z
# CHECK-NEXT:        %0 = affine.apply #map(%block_id_x)
# CHECK-NEXT:        %subview_18 = memref.subview %subview[%0, 0] [128, 128] [1, 1] : memref<128x128xf32, strided<[128, 1]>> to memref<128x128xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %subview_19 = memref.subview %subview_0[0, 0] [128, 256] [1, 1] : memref<128x256xf32, strided<[256, 1]>> to memref<128x256xf32, strided<[256, 1]>>
# CHECK-NEXT:        %subview_20 = memref.subview %subview_1[%0, 0] [128, 256] [1, 1] : memref<128x256xf32, strided<[256, 1]>> to memref<128x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        %thread_id_x = gpu.thread_id  x
# CHECK-NEXT:        %thread_id_y = gpu.thread_id  y
# CHECK-NEXT:        %thread_id_z = gpu.thread_id  z
# CHECK-NEXT:        %1 = affine.apply #map1(%thread_id_x)
# CHECK-NEXT:        %subview_21 = memref.subview %subview_18[%1, 0] [8, 128] [1, 1] : memref<128x128xf32, strided<[128, 1], offset: ?>> to memref<8x128xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %subview_22 = memref.subview %subview_19[0, 0] [128, 256] [1, 1] : memref<128x256xf32, strided<[256, 1]>> to memref<128x256xf32, strided<[256, 1]>>
# CHECK-NEXT:        %subview_23 = memref.subview %subview_20[%1, 0] [8, 256] [1, 1] : memref<128x256xf32, strided<[256, 1], offset: ?>> to memref<8x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        %c0_24 = arith.constant 0 : index
# CHECK-NEXT:        %c8_25 = arith.constant 8 : index
# CHECK-NEXT:        %c1_26 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg15 = %c0_24 to %c8_25 step %c1_26 {
# CHECK-NEXT:          %subview_27 = memref.subview %subview_21[%arg15, 0] [1, 128] [1, 1] : memref<8x128xf32, strided<[128, 1], offset: ?>> to memref<1x128xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:          %subview_28 = memref.subview %subview_22[0, 0] [128, 256] [1, 1] : memref<128x256xf32, strided<[256, 1]>> to memref<128x256xf32, strided<[256, 1]>>
# CHECK-NEXT:          %subview_29 = memref.subview %subview_23[%arg15, 0] [1, 256] [1, 1] : memref<8x256xf32, strided<[256, 1], offset: ?>> to memref<1x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:          %c0_30 = arith.constant 0 : index
# CHECK-NEXT:          %c128 = arith.constant 128 : index
# CHECK-NEXT:          %c1_31 = arith.constant 1 : index
# CHECK-NEXT:          scf.for %arg16 = %c0_30 to %c128 step %c1_31 {
# CHECK-NEXT:            %subview_32 = memref.subview %subview_27[0, %arg16] [1, 1] [1, 1] : memref<1x128xf32, strided<[128, 1], offset: ?>> to memref<1x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:            %subview_33 = memref.subview %subview_28[%arg16, 0] [1, 256] [1, 1] : memref<128x256xf32, strided<[256, 1]>> to memref<1x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:            %subview_34 = memref.subview %subview_29[0, 0] [1, 256] [1, 1] : memref<1x256xf32, strided<[256, 1], offset: ?>> to memref<1x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:            %c0_35 = arith.constant 0 : index
# CHECK-NEXT:            %c256 = arith.constant 256 : index
# CHECK-NEXT:            %c1_36 = arith.constant 1 : index
# CHECK-NEXT:            scf.for %arg17 = %c0_35 to %c256 step %c1_36 {
# CHECK-NEXT:              %subview_37 = memref.subview %subview_32[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[128, 1], offset: ?>> to memref<1x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:              %subview_38 = memref.subview %subview_33[0, %arg17] [1, 1] [1, 1] : memref<1x256xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:              %subview_39 = memref.subview %subview_34[0, %arg17] [1, 1] [1, 1] : memref<1x256xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:              linalg.matmul {__xtc_id_C_} ins(%subview_37, %subview_38 : memref<1x1xf32, strided<[128, 1], offset: ?>>, memref<1x1xf32, strided<[256, 1], offset: ?>>) outs(%subview_39 : memref<1x1xf32, strided<[256, 1], offset: ?>>)
# CHECK-NEXT:            } {"C/I[0]/J"}
# CHECK-NEXT:          } {"C/I[0]/K"}
# CHECK-NEXT:        } {"C/I[0]/I1"}
# CHECK-NEXT:        gpu.barrier
# CHECK-NEXT:        gpu.terminator
# CHECK-NEXT:      }
# CHECK-NEXT:      %subview_8 = memref.subview %arg0[128, 0] [384, 128] [1, 1] : memref<512x128xf32> to memref<384x128xf32, strided<[128, 1], offset: 16384>>
# CHECK-NEXT:      %subview_9 = memref.subview %arg1[0, 0] [128, 256] [1, 1] : memref<128x256xf32> to memref<128x256xf32, strided<[256, 1]>>
# CHECK-NEXT:      %subview_10 = memref.subview %arg2[128, 0] [384, 256] [1, 1] : memref<512x256xf32> to memref<384x256xf32, strided<[256, 1], offset: 32768>>
# CHECK-NEXT:      %c1_11 = arith.constant 1 : index
# CHECK-NEXT:      %c8 = arith.constant 8 : index
# CHECK-NEXT:      %c1_12 = arith.constant 1 : index
# CHECK-NEXT:      %c1_13 = arith.constant 1 : index
# CHECK-NEXT:      %c6 = arith.constant 6 : index
# CHECK-NEXT:      %c1_14 = arith.constant 1 : index
# CHECK-NEXT:      %c1_15 = arith.constant 1 : index
# CHECK-NEXT:      gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c6, %arg10 = %c1_14, %arg11 = %c1_15) threads(%arg6, %arg7, %arg8) in (%arg12 = %c8, %arg13 = %c1_12, %arg14 = %c1_13) {
# CHECK-NEXT:        %c0_16 = arith.constant 0 : index
# CHECK-NEXT:        %c0_17 = arith.constant 0 : index
# CHECK-NEXT:        %block_id_x = gpu.block_id  x
# CHECK-NEXT:        %block_id_y = gpu.block_id  y
# CHECK-NEXT:        %block_id_z = gpu.block_id  z
# CHECK-NEXT:        %0 = affine.apply #map2(%block_id_x)
# CHECK-NEXT:        %subview_18 = memref.subview %subview_8[%0, 0] [64, 128] [1, 1] : memref<384x128xf32, strided<[128, 1], offset: 16384>> to memref<64x128xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %subview_19 = memref.subview %subview_9[0, 0] [128, 256] [1, 1] : memref<128x256xf32, strided<[256, 1]>> to memref<128x256xf32, strided<[256, 1]>>
# CHECK-NEXT:        %subview_20 = memref.subview %subview_10[%0, 0] [64, 256] [1, 1] : memref<384x256xf32, strided<[256, 1], offset: 32768>> to memref<64x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        %thread_id_x = gpu.thread_id  x
# CHECK-NEXT:        %thread_id_y = gpu.thread_id  y
# CHECK-NEXT:        %thread_id_z = gpu.thread_id  z
# CHECK-NEXT:        %1 = affine.apply #map1(%thread_id_x)
# CHECK-NEXT:        %subview_21 = memref.subview %subview_18[%1, 0] [8, 128] [1, 1] : memref<64x128xf32, strided<[128, 1], offset: ?>> to memref<8x128xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %subview_22 = memref.subview %subview_19[0, 0] [128, 256] [1, 1] : memref<128x256xf32, strided<[256, 1]>> to memref<128x256xf32, strided<[256, 1]>>
# CHECK-NEXT:        %subview_23 = memref.subview %subview_20[%1, 0] [8, 256] [1, 1] : memref<64x256xf32, strided<[256, 1], offset: ?>> to memref<8x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        %c0_24 = arith.constant 0 : index
# CHECK-NEXT:        %c8_25 = arith.constant 8 : index
# CHECK-NEXT:        %c1_26 = arith.constant 1 : index
# CHECK-NEXT:        scf.for %arg15 = %c0_24 to %c8_25 step %c1_26 {
# CHECK-NEXT:          %subview_27 = memref.subview %subview_21[%arg15, 0] [1, 128] [1, 1] : memref<8x128xf32, strided<[128, 1], offset: ?>> to memref<1x128xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:          %subview_28 = memref.subview %subview_22[0, 0] [128, 256] [1, 1] : memref<128x256xf32, strided<[256, 1]>> to memref<128x256xf32, strided<[256, 1]>>
# CHECK-NEXT:          %subview_29 = memref.subview %subview_23[%arg15, 0] [1, 256] [1, 1] : memref<8x256xf32, strided<[256, 1], offset: ?>> to memref<1x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:          %c0_30 = arith.constant 0 : index
# CHECK-NEXT:          %c128 = arith.constant 128 : index
# CHECK-NEXT:          %c1_31 = arith.constant 1 : index
# CHECK-NEXT:          scf.for %arg16 = %c0_30 to %c128 step %c1_31 {
# CHECK-NEXT:            %subview_32 = memref.subview %subview_27[0, %arg16] [1, 1] [1, 1] : memref<1x128xf32, strided<[128, 1], offset: ?>> to memref<1x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:            %subview_33 = memref.subview %subview_28[%arg16, 0] [1, 256] [1, 1] : memref<128x256xf32, strided<[256, 1]>> to memref<1x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:            %subview_34 = memref.subview %subview_29[0, 0] [1, 256] [1, 1] : memref<1x256xf32, strided<[256, 1], offset: ?>> to memref<1x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:            %c0_35 = arith.constant 0 : index
# CHECK-NEXT:            %c256 = arith.constant 256 : index
# CHECK-NEXT:            %c1_36 = arith.constant 1 : index
# CHECK-NEXT:            scf.for %arg17 = %c0_35 to %c256 step %c1_36 {
# CHECK-NEXT:              %subview_37 = memref.subview %subview_32[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[128, 1], offset: ?>> to memref<1x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:              %subview_38 = memref.subview %subview_33[0, %arg17] [1, 1] [1, 1] : memref<1x256xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:              %subview_39 = memref.subview %subview_34[0, %arg17] [1, 1] [1, 1] : memref<1x256xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:              linalg.matmul {__xtc_id_C_} ins(%subview_37, %subview_38 : memref<1x1xf32, strided<[128, 1], offset: ?>>, memref<1x1xf32, strided<[256, 1], offset: ?>>) outs(%subview_39 : memref<1x1xf32, strided<[256, 1], offset: ?>>)
# CHECK-NEXT:            } {"C/I[1]/J"}
# CHECK-NEXT:          } {"C/I[1]/K"}
# CHECK-NEXT:        } {"C/I[1]/I1"}
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
# CHECK-NEXT:    - %0 : 512x128xfloat32
# CHECK-NEXT:    - %1 : 128x256xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %2 : 512x256xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: matmul(%0, %1) {name = 'C'} : [512x128xfloat32, 128x256xfloat32] -> [512x256xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  CODE: 0

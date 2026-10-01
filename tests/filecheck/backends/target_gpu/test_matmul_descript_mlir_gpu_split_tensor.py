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

impl = Backend(gb.graph, use_tensor_dialect=True)

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
            "I#128": {"gpu_thread": 0},
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
    dump_file="matmul_descript_gpu_split_tensor",
    print_source_ir=True,
    print_transformed_ir=True,
    print_bufferization_ir=True,
)
module = comp.compile(sched)

evaluator = module.get_evaluator(validate=True)
results, code, error = evaluator.evaluate()
print(f"CODE: {code}")

# CHECK:       // -----// IR Dump Before transform //----- //
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: tensor<512x128xf32> {llvm.noalias, memref.on_device}, %arg1: tensor<128x256xf32> {llvm.noalias, memref.on_device}, %arg2: memref<512x256xf32> {llvm.noalias, memref.on_device}) {
# CHECK-NEXT:      %0 = tensor.empty() : tensor<512x256xf32>
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %1 = linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%0 : tensor<512x256xf32>) -> tensor<512x256xf32>
# CHECK-NEXT:      %2 = linalg.matmul {__xtc_id_C_} ins(%arg0, %arg1 : tensor<512x128xf32>, tensor<128x256xf32>) outs(%1 : tensor<512x256xf32>) -> tensor<512x256xf32>
# CHECK-NEXT:      bufferization.materialize_in_destination %2 in restrict writable %arg2 : (tensor<512x256xf32>, memref<512x256xf32>) -> ()
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:      transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_post_bufferize(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = transform.structured.match attributes {"C/I[0]/I", mapping = [#gpu.block<x>]} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %1 = transform.gpu.map_forall_to_blocks %0 generate_gpu_launch : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %2 = transform.gpu.map_nested_forall_to_threads %1 block_dims = [16, 1, 1] : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %3 = transform.structured.match attributes {"C/I[1]/I", mapping = [#gpu.block<x>]} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %4 = transform.gpu.map_forall_to_blocks %3 generate_gpu_launch : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %5 = transform.gpu.map_nested_forall_to_threads %4 block_dims = [16, 1, 1] : (!transform.any_op) -> !transform.any_op
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
# CHECK-NEXT:      %tiled_op_10, %forall_op_11 = transform.structured.tile_using_forall %3#1 tile_sizes [128, 0, 0](mapping = [#gpu.block<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %forall_op_11 "C/I[1]/I" : !transform.any_op
# CHECK-NEXT:      %tiled_op_12, %forall_op_13 = transform.structured.tile_using_forall %tiled_op_10 tile_sizes [8, 0, 0](mapping = [#gpu.thread<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %forall_op_13 "C/I[1]/I0" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_14, %loops_15 = transform.structured.tile_using_for %tiled_op_12 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_15 "C/I[1]/I1" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_16, %loops_17 = transform.structured.tile_using_for %tiled_linalg_op_14 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_17 "C/I[1]/K" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_18, %loops_19 = transform.structured.tile_using_for %tiled_linalg_op_16 tile_sizes [0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_19 "C/I[1]/J" : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0) -> (d0 * 128)>
# CHECK-NEXT:  #map1 = affine_map<(d0) -> (d0 * 8)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: tensor<512x128xf32> {llvm.noalias, memref.on_device}, %arg1: tensor<128x256xf32> {llvm.noalias, memref.on_device}, %arg2: memref<512x256xf32> {llvm.noalias, memref.on_device}) {
# CHECK-NEXT:      %0 = tensor.empty() : tensor<512x256xf32>
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c512 = arith.constant 512 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      %1 = scf.for %arg3 = %c0 to %c512 step %c1 iter_args(%arg4 = %0) -> (tensor<512x256xf32>) {
# CHECK-NEXT:        %extracted_slice_6 = tensor.extract_slice %arg4[%arg3, 0] [1, 256] [1, 1] : tensor<512x256xf32> to tensor<1x256xf32>
# CHECK-NEXT:        %c0_7 = arith.constant 0 : index
# CHECK-NEXT:        %c256 = arith.constant 256 : index
# CHECK-NEXT:        %c1_8 = arith.constant 1 : index
# CHECK-NEXT:        %4 = scf.for %arg5 = %c0_7 to %c256 step %c1_8 iter_args(%arg6 = %extracted_slice_6) -> (tensor<1x256xf32>) {
# CHECK-NEXT:          %extracted_slice_10 = tensor.extract_slice %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x256xf32> to tensor<1x1xf32>
# CHECK-NEXT:          %5 = linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%extracted_slice_10 : tensor<1x1xf32>) -> tensor<1x1xf32>
# CHECK-NEXT:          %inserted_slice_11 = tensor.insert_slice %5 into %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x256xf32>
# CHECK-NEXT:          scf.yield %inserted_slice_11 : tensor<1x256xf32>
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:        %inserted_slice_9 = tensor.insert_slice %4 into %arg4[%arg3, 0] [1, 256] [1, 1] : tensor<1x256xf32> into tensor<512x256xf32>
# CHECK-NEXT:        scf.yield %inserted_slice_9 : tensor<512x256xf32>
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %extracted_slice = tensor.extract_slice %arg0[0, 0] [128, 128] [1, 1] : tensor<512x128xf32> to tensor<128x128xf32>
# CHECK-NEXT:      %extracted_slice_0 = tensor.extract_slice %arg1[0, 0] [128, 256] [1, 1] : tensor<128x256xf32> to tensor<128x256xf32>
# CHECK-NEXT:      %extracted_slice_1 = tensor.extract_slice %1[0, 0] [128, 256] [1, 1] : tensor<512x256xf32> to tensor<128x256xf32>
# CHECK-NEXT:      %2 = scf.forall (%arg3) in (1) shared_outs(%arg4 = %extracted_slice_1) -> (tensor<128x256xf32>) {
# CHECK-NEXT:        %4 = affine.apply #map(%arg3)
# CHECK-NEXT:        %extracted_slice_6 = tensor.extract_slice %extracted_slice[%4, 0] [128, 128] [1, 1] : tensor<128x128xf32> to tensor<128x128xf32>
# CHECK-NEXT:        %extracted_slice_7 = tensor.extract_slice %extracted_slice_0[0, 0] [128, 256] [1, 1] : tensor<128x256xf32> to tensor<128x256xf32>
# CHECK-NEXT:        %extracted_slice_8 = tensor.extract_slice %arg4[%4, 0] [128, 256] [1, 1] : tensor<128x256xf32> to tensor<128x256xf32>
# CHECK-NEXT:        %5 = scf.forall (%arg5) in (16) shared_outs(%arg6 = %extracted_slice_8) -> (tensor<128x256xf32>) {
# CHECK-NEXT:          %6 = affine.apply #map1(%arg5)
# CHECK-NEXT:          %extracted_slice_9 = tensor.extract_slice %extracted_slice_6[%6, 0] [8, 128] [1, 1] : tensor<128x128xf32> to tensor<8x128xf32>
# CHECK-NEXT:          %extracted_slice_10 = tensor.extract_slice %extracted_slice_7[0, 0] [128, 256] [1, 1] : tensor<128x256xf32> to tensor<128x256xf32>
# CHECK-NEXT:          %extracted_slice_11 = tensor.extract_slice %arg6[%6, 0] [8, 256] [1, 1] : tensor<128x256xf32> to tensor<8x256xf32>
# CHECK-NEXT:          %c0_12 = arith.constant 0 : index
# CHECK-NEXT:          %c8 = arith.constant 8 : index
# CHECK-NEXT:          %c1_13 = arith.constant 1 : index
# CHECK-NEXT:          %7 = scf.for %arg7 = %c0_12 to %c8 step %c1_13 iter_args(%arg8 = %extracted_slice_11) -> (tensor<8x256xf32>) {
# CHECK-NEXT:            %extracted_slice_14 = tensor.extract_slice %extracted_slice_9[%arg7, 0] [1, 128] [1, 1] : tensor<8x128xf32> to tensor<1x128xf32>
# CHECK-NEXT:            %extracted_slice_15 = tensor.extract_slice %extracted_slice_10[0, 0] [128, 256] [1, 1] : tensor<128x256xf32> to tensor<128x256xf32>
# CHECK-NEXT:            %extracted_slice_16 = tensor.extract_slice %arg8[%arg7, 0] [1, 256] [1, 1] : tensor<8x256xf32> to tensor<1x256xf32>
# CHECK-NEXT:            %c0_17 = arith.constant 0 : index
# CHECK-NEXT:            %c128 = arith.constant 128 : index
# CHECK-NEXT:            %c1_18 = arith.constant 1 : index
# CHECK-NEXT:            %8 = scf.for %arg9 = %c0_17 to %c128 step %c1_18 iter_args(%arg10 = %extracted_slice_16) -> (tensor<1x256xf32>) {
# CHECK-NEXT:              %extracted_slice_20 = tensor.extract_slice %extracted_slice_14[0, %arg9] [1, 1] [1, 1] : tensor<1x128xf32> to tensor<1x1xf32>
# CHECK-NEXT:              %extracted_slice_21 = tensor.extract_slice %extracted_slice_15[%arg9, 0] [1, 256] [1, 1] : tensor<128x256xf32> to tensor<1x256xf32>
# CHECK-NEXT:              %extracted_slice_22 = tensor.extract_slice %arg10[0, 0] [1, 256] [1, 1] : tensor<1x256xf32> to tensor<1x256xf32>
# CHECK-NEXT:              %c0_23 = arith.constant 0 : index
# CHECK-NEXT:              %c256 = arith.constant 256 : index
# CHECK-NEXT:              %c1_24 = arith.constant 1 : index
# CHECK-NEXT:              %9 = scf.for %arg11 = %c0_23 to %c256 step %c1_24 iter_args(%arg12 = %extracted_slice_22) -> (tensor<1x256xf32>) {
# CHECK-NEXT:                %extracted_slice_26 = tensor.extract_slice %extracted_slice_20[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> to tensor<1x1xf32>
# CHECK-NEXT:                %extracted_slice_27 = tensor.extract_slice %extracted_slice_21[0, %arg11] [1, 1] [1, 1] : tensor<1x256xf32> to tensor<1x1xf32>
# CHECK-NEXT:                %extracted_slice_28 = tensor.extract_slice %arg12[0, %arg11] [1, 1] [1, 1] : tensor<1x256xf32> to tensor<1x1xf32>
# CHECK-NEXT:                %10 = linalg.matmul {__xtc_id_C_} ins(%extracted_slice_26, %extracted_slice_27 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%extracted_slice_28 : tensor<1x1xf32>) -> tensor<1x1xf32>
# CHECK-NEXT:                %inserted_slice_29 = tensor.insert_slice %10 into %arg12[0, %arg11] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x256xf32>
# CHECK-NEXT:                scf.yield %inserted_slice_29 : tensor<1x256xf32>
# CHECK-NEXT:              } {"C/I[0]/J"}
# CHECK-NEXT:              %inserted_slice_25 = tensor.insert_slice %9 into %arg10[0, 0] [1, 256] [1, 1] : tensor<1x256xf32> into tensor<1x256xf32>
# CHECK-NEXT:              scf.yield %inserted_slice_25 : tensor<1x256xf32>
# CHECK-NEXT:            } {"C/I[0]/K"}
# CHECK-NEXT:            %inserted_slice_19 = tensor.insert_slice %8 into %arg8[%arg7, 0] [1, 256] [1, 1] : tensor<1x256xf32> into tensor<8x256xf32>
# CHECK-NEXT:            scf.yield %inserted_slice_19 : tensor<8x256xf32>
# CHECK-NEXT:          } {"C/I[0]/I1"}
# CHECK-NEXT:          scf.forall.in_parallel {
# CHECK-NEXT:            tensor.parallel_insert_slice %7 into %arg6[%6, 0] [8, 256] [1, 1] : tensor<8x256xf32> into tensor<128x256xf32>
# CHECK-NEXT:          }
# CHECK-NEXT:        } {"C/I[0]/I0", mapping = [#gpu.thread<x>]}
# CHECK-NEXT:        scf.forall.in_parallel {
# CHECK-NEXT:          tensor.parallel_insert_slice %5 into %arg4[%4, 0] [128, 256] [1, 1] : tensor<128x256xf32> into tensor<128x256xf32>
# CHECK-NEXT:        }
# CHECK-NEXT:      } {"C/I[0]/I", mapping = [#gpu.block<x>]}
# CHECK-NEXT:      %inserted_slice = tensor.insert_slice %2 into %1[0, 0] [128, 256] [1, 1] : tensor<128x256xf32> into tensor<512x256xf32>
# CHECK-NEXT:      %extracted_slice_2 = tensor.extract_slice %arg0[128, 0] [384, 128] [1, 1] : tensor<512x128xf32> to tensor<384x128xf32>
# CHECK-NEXT:      %extracted_slice_3 = tensor.extract_slice %arg1[0, 0] [128, 256] [1, 1] : tensor<128x256xf32> to tensor<128x256xf32>
# CHECK-NEXT:      %extracted_slice_4 = tensor.extract_slice %inserted_slice[128, 0] [384, 256] [1, 1] : tensor<512x256xf32> to tensor<384x256xf32>
# CHECK-NEXT:      %3 = scf.forall (%arg3) in (3) shared_outs(%arg4 = %extracted_slice_4) -> (tensor<384x256xf32>) {
# CHECK-NEXT:        %4 = affine.apply #map(%arg3)
# CHECK-NEXT:        %extracted_slice_6 = tensor.extract_slice %extracted_slice_2[%4, 0] [128, 128] [1, 1] : tensor<384x128xf32> to tensor<128x128xf32>
# CHECK-NEXT:        %extracted_slice_7 = tensor.extract_slice %extracted_slice_3[0, 0] [128, 256] [1, 1] : tensor<128x256xf32> to tensor<128x256xf32>
# CHECK-NEXT:        %extracted_slice_8 = tensor.extract_slice %arg4[%4, 0] [128, 256] [1, 1] : tensor<384x256xf32> to tensor<128x256xf32>
# CHECK-NEXT:        %5 = scf.forall (%arg5) in (16) shared_outs(%arg6 = %extracted_slice_8) -> (tensor<128x256xf32>) {
# CHECK-NEXT:          %6 = affine.apply #map1(%arg5)
# CHECK-NEXT:          %extracted_slice_9 = tensor.extract_slice %extracted_slice_6[%6, 0] [8, 128] [1, 1] : tensor<128x128xf32> to tensor<8x128xf32>
# CHECK-NEXT:          %extracted_slice_10 = tensor.extract_slice %extracted_slice_7[0, 0] [128, 256] [1, 1] : tensor<128x256xf32> to tensor<128x256xf32>
# CHECK-NEXT:          %extracted_slice_11 = tensor.extract_slice %arg6[%6, 0] [8, 256] [1, 1] : tensor<128x256xf32> to tensor<8x256xf32>
# CHECK-NEXT:          %c0_12 = arith.constant 0 : index
# CHECK-NEXT:          %c8 = arith.constant 8 : index
# CHECK-NEXT:          %c1_13 = arith.constant 1 : index
# CHECK-NEXT:          %7 = scf.for %arg7 = %c0_12 to %c8 step %c1_13 iter_args(%arg8 = %extracted_slice_11) -> (tensor<8x256xf32>) {
# CHECK-NEXT:            %extracted_slice_14 = tensor.extract_slice %extracted_slice_9[%arg7, 0] [1, 128] [1, 1] : tensor<8x128xf32> to tensor<1x128xf32>
# CHECK-NEXT:            %extracted_slice_15 = tensor.extract_slice %extracted_slice_10[0, 0] [128, 256] [1, 1] : tensor<128x256xf32> to tensor<128x256xf32>
# CHECK-NEXT:            %extracted_slice_16 = tensor.extract_slice %arg8[%arg7, 0] [1, 256] [1, 1] : tensor<8x256xf32> to tensor<1x256xf32>
# CHECK-NEXT:            %c0_17 = arith.constant 0 : index
# CHECK-NEXT:            %c128 = arith.constant 128 : index
# CHECK-NEXT:            %c1_18 = arith.constant 1 : index
# CHECK-NEXT:            %8 = scf.for %arg9 = %c0_17 to %c128 step %c1_18 iter_args(%arg10 = %extracted_slice_16) -> (tensor<1x256xf32>) {
# CHECK-NEXT:              %extracted_slice_20 = tensor.extract_slice %extracted_slice_14[0, %arg9] [1, 1] [1, 1] : tensor<1x128xf32> to tensor<1x1xf32>
# CHECK-NEXT:              %extracted_slice_21 = tensor.extract_slice %extracted_slice_15[%arg9, 0] [1, 256] [1, 1] : tensor<128x256xf32> to tensor<1x256xf32>
# CHECK-NEXT:              %extracted_slice_22 = tensor.extract_slice %arg10[0, 0] [1, 256] [1, 1] : tensor<1x256xf32> to tensor<1x256xf32>
# CHECK-NEXT:              %c0_23 = arith.constant 0 : index
# CHECK-NEXT:              %c256 = arith.constant 256 : index
# CHECK-NEXT:              %c1_24 = arith.constant 1 : index
# CHECK-NEXT:              %9 = scf.for %arg11 = %c0_23 to %c256 step %c1_24 iter_args(%arg12 = %extracted_slice_22) -> (tensor<1x256xf32>) {
# CHECK-NEXT:                %extracted_slice_26 = tensor.extract_slice %extracted_slice_20[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> to tensor<1x1xf32>
# CHECK-NEXT:                %extracted_slice_27 = tensor.extract_slice %extracted_slice_21[0, %arg11] [1, 1] [1, 1] : tensor<1x256xf32> to tensor<1x1xf32>
# CHECK-NEXT:                %extracted_slice_28 = tensor.extract_slice %arg12[0, %arg11] [1, 1] [1, 1] : tensor<1x256xf32> to tensor<1x1xf32>
# CHECK-NEXT:                %10 = linalg.matmul {__xtc_id_C_} ins(%extracted_slice_26, %extracted_slice_27 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%extracted_slice_28 : tensor<1x1xf32>) -> tensor<1x1xf32>
# CHECK-NEXT:                %inserted_slice_29 = tensor.insert_slice %10 into %arg12[0, %arg11] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x256xf32>
# CHECK-NEXT:                scf.yield %inserted_slice_29 : tensor<1x256xf32>
# CHECK-NEXT:              } {"C/I[1]/J"}
# CHECK-NEXT:              %inserted_slice_25 = tensor.insert_slice %9 into %arg10[0, 0] [1, 256] [1, 1] : tensor<1x256xf32> into tensor<1x256xf32>
# CHECK-NEXT:              scf.yield %inserted_slice_25 : tensor<1x256xf32>
# CHECK-NEXT:            } {"C/I[1]/K"}
# CHECK-NEXT:            %inserted_slice_19 = tensor.insert_slice %8 into %arg8[%arg7, 0] [1, 256] [1, 1] : tensor<1x256xf32> into tensor<8x256xf32>
# CHECK-NEXT:            scf.yield %inserted_slice_19 : tensor<8x256xf32>
# CHECK-NEXT:          } {"C/I[1]/I1"}
# CHECK-NEXT:          scf.forall.in_parallel {
# CHECK-NEXT:            tensor.parallel_insert_slice %7 into %arg6[%6, 0] [8, 256] [1, 1] : tensor<8x256xf32> into tensor<128x256xf32>
# CHECK-NEXT:          }
# CHECK-NEXT:        } {"C/I[1]/I0", mapping = [#gpu.thread<x>]}
# CHECK-NEXT:        scf.forall.in_parallel {
# CHECK-NEXT:          tensor.parallel_insert_slice %5 into %arg4[%4, 0] [128, 256] [1, 1] : tensor<128x256xf32> into tensor<384x256xf32>
# CHECK-NEXT:        }
# CHECK-NEXT:      } {"C/I[1]/I", mapping = [#gpu.block<x>]}
# CHECK-NEXT:      %inserted_slice_5 = tensor.insert_slice %3 into %inserted_slice[128, 0] [384, 256] [1, 1] : tensor<384x256xf32> into tensor<512x256xf32>
# CHECK-NEXT:      bufferization.materialize_in_destination %inserted_slice_5 in restrict writable %arg2 : (tensor<512x256xf32>, memref<512x256xf32>) -> ()
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:      transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_post_bufferize(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = transform.structured.match attributes {"C/I[0]/I", mapping = [#gpu.block<x>]} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %1 = transform.gpu.map_forall_to_blocks %0 generate_gpu_launch : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %2 = transform.gpu.map_nested_forall_to_threads %1 block_dims = [16, 1, 1] : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %3 = transform.structured.match attributes {"C/I[1]/I", mapping = [#gpu.block<x>]} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %4 = transform.gpu.map_forall_to_blocks %3 generate_gpu_launch : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %5 = transform.gpu.map_nested_forall_to_threads %4 block_dims = [16, 1, 1] : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After Tensor Lowering //----- //
# CHECK-NEXT:  #map = affine_map<(d0) -> (d0 * 8)>
# CHECK-NEXT:  #map1 = affine_map<(d0) -> (d0 * 128)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: memref<512x128xf32> {llvm.noalias, memref.on_device}, %arg1: memref<128x256xf32> {llvm.noalias, memref.on_device}, %arg2: memref<512x256xf32> {llvm.noalias, memref.on_device}) {
# CHECK-NEXT:      %c128 = arith.constant 128 : index
# CHECK-NEXT:      %c8 = arith.constant 8 : index
# CHECK-NEXT:      %c256 = arith.constant 256 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      %c512 = arith.constant 512 : index
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %0 = scf.for %arg3 = %c0 to %c512 step %c1 iter_args(%arg4 = %arg2) -> (memref<512x256xf32>) {
# CHECK-NEXT:        %subview_17 = memref.subview %arg4[%arg3, 0] [1, 256] [1, 1] : memref<512x256xf32> to memref<1x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        %1 = scf.for %arg5 = %c0 to %c256 step %c1 iter_args(%arg6 = %subview_17) -> (memref<1x256xf32, strided<[256, 1], offset: ?>>) {
# CHECK-NEXT:          %subview_19 = memref.subview %arg6[0, %arg5] [1, 1] [1, 1] : memref<1x256xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:          linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%subview_19 : memref<1x1xf32, strided<[256, 1], offset: ?>>)
# CHECK-NEXT:          %subview_20 = memref.subview %arg6[0, %arg5] [1, 1] [1, 1] : memref<1x256xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:          memref.copy %subview_19, %subview_20 : memref<1x1xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:          scf.yield %arg6 : memref<1x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:        %subview_18 = memref.subview %arg4[%arg3, 0] [1, 256] [1, 1] : memref<512x256xf32> to memref<1x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        memref.copy %1, %subview_18 : memref<1x256xf32, strided<[256, 1], offset: ?>> to memref<1x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        scf.yield %arg4 : memref<512x256xf32>
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      %subview = memref.subview %arg0[0, 0] [128, 128] [1, 1] : memref<512x128xf32> to memref<128x128xf32, strided<[128, 1]>>
# CHECK-NEXT:      %subview_0 = memref.subview %0[0, 0] [128, 256] [1, 1] : memref<512x256xf32> to memref<128x256xf32, strided<[256, 1]>>
# CHECK-NEXT:      %c1_1 = arith.constant 1 : index
# CHECK-NEXT:      %c16 = arith.constant 16 : index
# CHECK-NEXT:      %c1_2 = arith.constant 1 : index
# CHECK-NEXT:      %c1_3 = arith.constant 1 : index
# CHECK-NEXT:      %c1_4 = arith.constant 1 : index
# CHECK-NEXT:      %c1_5 = arith.constant 1 : index
# CHECK-NEXT:      %c1_6 = arith.constant 1 : index
# CHECK-NEXT:      gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c1_4, %arg10 = %c1_5, %arg11 = %c1_6) threads(%arg6, %arg7, %arg8) in (%arg12 = %c16, %arg13 = %c1_2, %arg14 = %c1_3) {
# CHECK-NEXT:        %c0_17 = arith.constant 0 : index
# CHECK-NEXT:        %c0_18 = arith.constant 0 : index
# CHECK-NEXT:        %block_id_x = gpu.block_id  x
# CHECK-NEXT:        %block_id_y = gpu.block_id  y
# CHECK-NEXT:        %block_id_z = gpu.block_id  z
# CHECK-NEXT:        %thread_id_x = gpu.thread_id  x
# CHECK-NEXT:        %thread_id_y = gpu.thread_id  y
# CHECK-NEXT:        %thread_id_z = gpu.thread_id  z
# CHECK-NEXT:        %1 = affine.apply #map(%thread_id_x)
# CHECK-NEXT:        %subview_19 = memref.subview %subview[%1, 0] [8, 128] [1, 1] : memref<128x128xf32, strided<[128, 1]>> to memref<8x128xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %subview_20 = memref.subview %subview_0[%1, 0] [8, 256] [1, 1] : memref<128x256xf32, strided<[256, 1]>> to memref<8x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        %2 = scf.for %arg15 = %c0 to %c8 step %c1 iter_args(%arg16 = %subview_20) -> (memref<8x256xf32, strided<[256, 1], offset: ?>>) {
# CHECK-NEXT:          %subview_23 = memref.subview %subview_19[%arg15, 0] [1, 128] [1, 1] : memref<8x128xf32, strided<[128, 1], offset: ?>> to memref<1x128xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:          %subview_24 = memref.subview %arg16[%arg15, 0] [1, 256] [1, 1] : memref<8x256xf32, strided<[256, 1], offset: ?>> to memref<1x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:          %3 = scf.for %arg17 = %c0 to %c128 step %c1 iter_args(%arg18 = %subview_24) -> (memref<1x256xf32, strided<[256, 1], offset: ?>>) {
# CHECK-NEXT:            %subview_26 = memref.subview %subview_23[0, %arg17] [1, 1] [1, 1] : memref<1x128xf32, strided<[128, 1], offset: ?>> to memref<1x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:            %subview_27 = memref.subview %arg1[%arg17, 0] [1, 256] [1, 1] : memref<128x256xf32> to memref<1x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:            %4 = scf.for %arg19 = %c0 to %c256 step %c1 iter_args(%arg20 = %arg18) -> (memref<1x256xf32, strided<[256, 1], offset: ?>>) {
# CHECK-NEXT:              %subview_28 = memref.subview %subview_27[0, %arg19] [1, 1] [1, 1] : memref<1x256xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:              %subview_29 = memref.subview %arg20[0, %arg19] [1, 1] [1, 1] : memref<1x256xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:              linalg.matmul {__xtc_id_C_} ins(%subview_26, %subview_28 : memref<1x1xf32, strided<[128, 1], offset: ?>>, memref<1x1xf32, strided<[256, 1], offset: ?>>) outs(%subview_29 : memref<1x1xf32, strided<[256, 1], offset: ?>>)
# CHECK-NEXT:              %subview_30 = memref.subview %arg20[0, %arg19] [1, 1] [1, 1] : memref<1x256xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:              memref.copy %subview_29, %subview_30 : memref<1x1xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:              scf.yield %arg20 : memref<1x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:            } {"C/I[0]/J"}
# CHECK-NEXT:            scf.yield %4 : memref<1x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:          } {"C/I[0]/K"}
# CHECK-NEXT:          %subview_25 = memref.subview %arg16[%arg15, 0] [1, 256] [1, 1] : memref<8x256xf32, strided<[256, 1], offset: ?>> to memref<1x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:          memref.copy %3, %subview_25 : memref<1x256xf32, strided<[256, 1], offset: ?>> to memref<1x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:          scf.yield %arg16 : memref<8x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        } {"C/I[0]/I1"}
# CHECK-NEXT:        %subview_21 = memref.subview %subview_0[%1, 0] [8, 256] [1, 1] : memref<128x256xf32, strided<[256, 1]>> to memref<8x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        memref.copy %2, %subview_21 : memref<8x256xf32, strided<[256, 1], offset: ?>> to memref<8x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        gpu.barrier
# CHECK-NEXT:        %subview_22 = memref.subview %subview_0[0, 0] [128, 256] [1, 1] : memref<128x256xf32, strided<[256, 1]>> to memref<128x256xf32, strided<[256, 1]>>
# CHECK-NEXT:        memref.copy %subview_0, %subview_22 : memref<128x256xf32, strided<[256, 1]>> to memref<128x256xf32, strided<[256, 1]>>
# CHECK-NEXT:        gpu.terminator
# CHECK-NEXT:      }
# CHECK-NEXT:      %subview_7 = memref.subview %0[0, 0] [128, 256] [1, 1] : memref<512x256xf32> to memref<128x256xf32, strided<[256, 1]>>
# CHECK-NEXT:      memref.copy %subview_0, %subview_7 : memref<128x256xf32, strided<[256, 1]>> to memref<128x256xf32, strided<[256, 1]>>
# CHECK-NEXT:      %subview_8 = memref.subview %arg0[128, 0] [384, 128] [1, 1] : memref<512x128xf32> to memref<384x128xf32, strided<[128, 1], offset: 16384>>
# CHECK-NEXT:      %subview_9 = memref.subview %0[128, 0] [384, 256] [1, 1] : memref<512x256xf32> to memref<384x256xf32, strided<[256, 1], offset: 32768>>
# CHECK-NEXT:      %c1_10 = arith.constant 1 : index
# CHECK-NEXT:      %c16_11 = arith.constant 16 : index
# CHECK-NEXT:      %c1_12 = arith.constant 1 : index
# CHECK-NEXT:      %c1_13 = arith.constant 1 : index
# CHECK-NEXT:      %c3 = arith.constant 3 : index
# CHECK-NEXT:      %c1_14 = arith.constant 1 : index
# CHECK-NEXT:      %c1_15 = arith.constant 1 : index
# CHECK-NEXT:      gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c3, %arg10 = %c1_14, %arg11 = %c1_15) threads(%arg6, %arg7, %arg8) in (%arg12 = %c16_11, %arg13 = %c1_12, %arg14 = %c1_13) {
# CHECK-NEXT:        %c0_17 = arith.constant 0 : index
# CHECK-NEXT:        %c0_18 = arith.constant 0 : index
# CHECK-NEXT:        %block_id_x = gpu.block_id  x
# CHECK-NEXT:        %block_id_y = gpu.block_id  y
# CHECK-NEXT:        %block_id_z = gpu.block_id  z
# CHECK-NEXT:        %1 = affine.apply #map1(%block_id_x)
# CHECK-NEXT:        %subview_19 = memref.subview %subview_8[%1, 0] [128, 128] [1, 1] : memref<384x128xf32, strided<[128, 1], offset: 16384>> to memref<128x128xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %subview_20 = memref.subview %subview_9[%1, 0] [128, 256] [1, 1] : memref<384x256xf32, strided<[256, 1], offset: 32768>> to memref<128x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        %thread_id_x = gpu.thread_id  x
# CHECK-NEXT:        %thread_id_y = gpu.thread_id  y
# CHECK-NEXT:        %thread_id_z = gpu.thread_id  z
# CHECK-NEXT:        %2 = affine.apply #map(%thread_id_x)
# CHECK-NEXT:        %subview_21 = memref.subview %subview_19[%2, 0] [8, 128] [1, 1] : memref<128x128xf32, strided<[128, 1], offset: ?>> to memref<8x128xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:        %subview_22 = memref.subview %subview_20[%2, 0] [8, 256] [1, 1] : memref<128x256xf32, strided<[256, 1], offset: ?>> to memref<8x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        %3 = scf.for %arg15 = %c0 to %c8 step %c1 iter_args(%arg16 = %subview_22) -> (memref<8x256xf32, strided<[256, 1], offset: ?>>) {
# CHECK-NEXT:          %subview_25 = memref.subview %subview_21[%arg15, 0] [1, 128] [1, 1] : memref<8x128xf32, strided<[128, 1], offset: ?>> to memref<1x128xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:          %subview_26 = memref.subview %arg16[%arg15, 0] [1, 256] [1, 1] : memref<8x256xf32, strided<[256, 1], offset: ?>> to memref<1x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:          %4 = scf.for %arg17 = %c0 to %c128 step %c1 iter_args(%arg18 = %subview_26) -> (memref<1x256xf32, strided<[256, 1], offset: ?>>) {
# CHECK-NEXT:            %subview_28 = memref.subview %subview_25[0, %arg17] [1, 1] [1, 1] : memref<1x128xf32, strided<[128, 1], offset: ?>> to memref<1x1xf32, strided<[128, 1], offset: ?>>
# CHECK-NEXT:            %subview_29 = memref.subview %arg1[%arg17, 0] [1, 256] [1, 1] : memref<128x256xf32> to memref<1x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:            %5 = scf.for %arg19 = %c0 to %c256 step %c1 iter_args(%arg20 = %arg18) -> (memref<1x256xf32, strided<[256, 1], offset: ?>>) {
# CHECK-NEXT:              %subview_30 = memref.subview %subview_29[0, %arg19] [1, 1] [1, 1] : memref<1x256xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:              %subview_31 = memref.subview %arg20[0, %arg19] [1, 1] [1, 1] : memref<1x256xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:              linalg.matmul {__xtc_id_C_} ins(%subview_28, %subview_30 : memref<1x1xf32, strided<[128, 1], offset: ?>>, memref<1x1xf32, strided<[256, 1], offset: ?>>) outs(%subview_31 : memref<1x1xf32, strided<[256, 1], offset: ?>>)
# CHECK-NEXT:              %subview_32 = memref.subview %arg20[0, %arg19] [1, 1] [1, 1] : memref<1x256xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:              memref.copy %subview_31, %subview_32 : memref<1x1xf32, strided<[256, 1], offset: ?>> to memref<1x1xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:              scf.yield %arg20 : memref<1x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:            } {"C/I[1]/J"}
# CHECK-NEXT:            scf.yield %5 : memref<1x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:          } {"C/I[1]/K"}
# CHECK-NEXT:          %subview_27 = memref.subview %arg16[%arg15, 0] [1, 256] [1, 1] : memref<8x256xf32, strided<[256, 1], offset: ?>> to memref<1x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:          memref.copy %4, %subview_27 : memref<1x256xf32, strided<[256, 1], offset: ?>> to memref<1x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:          scf.yield %arg16 : memref<8x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        } {"C/I[1]/I1"}
# CHECK-NEXT:        %subview_23 = memref.subview %subview_20[%2, 0] [8, 256] [1, 1] : memref<128x256xf32, strided<[256, 1], offset: ?>> to memref<8x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        memref.copy %3, %subview_23 : memref<8x256xf32, strided<[256, 1], offset: ?>> to memref<8x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        gpu.barrier
# CHECK-NEXT:        %subview_24 = memref.subview %subview_9[%1, 0] [128, 256] [1, 1] : memref<384x256xf32, strided<[256, 1], offset: 32768>> to memref<128x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        memref.copy %subview_20, %subview_24 : memref<128x256xf32, strided<[256, 1], offset: ?>> to memref<128x256xf32, strided<[256, 1], offset: ?>>
# CHECK-NEXT:        gpu.terminator
# CHECK-NEXT:      }
# CHECK-NEXT:      %subview_16 = memref.subview %0[128, 0] [384, 256] [1, 1] : memref<512x256xf32> to memref<384x256xf32, strided<[256, 1], offset: 32768>>
# CHECK-NEXT:      memref.copy %subview_9, %subview_16 : memref<384x256xf32, strided<[256, 1], offset: 32768>> to memref<384x256xf32, strided<[256, 1], offset: 32768>>
# CHECK-NEXT:      memref.copy %0, %arg2 : memref<512x256xf32> to memref<512x256xf32>
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

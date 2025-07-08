
from xtc.schedules.ttile.scheme import build_scheme_from_str
from xtc.schedules.ttile.computation import Computation, Computation_spec
from xtc.schedules.ttile.archi import laptop_guillaume_machine, pinocchio_machine

from xtc.schedules.ttile.scheme_to_xdsltransform import convert_scheme_to_xdsl_transform_input, convert_scheme_to_loopschedule
from xtc.schedules.ttile.scheme_to_xdsltransform import build_xdsl_module_string_matmul, build_xdsl_module_string_conv
from xtc.schedules.ttile.scheme_to_xdsltransform import launch_and_measure_scheme

# Test of "convert_scheme_to_xdsl_transform_input"
def test_convert_scheme_to_xdsl_transform_input_1():
  comp = Computation(Computation_spec.CONV, 4)
  str_scheme = "[V (F, 16); U (F, 4); U (X, 2); U (Y, 2); U (C, 4); T (C, 16); (Hoist_var([C])); (T (F, 2)); (T (X, 7)); (T (Y, 14)); (T (X, 2));  (T (W, 3)); (T (H, 3))]"
  
  scheme = build_scheme_from_str(str_scheme)
  [parallel_dims, reduction_dims, dtiles_name,
    ltilesizes, llooporder, ldim_vectorize, l_dim_parallelized,
    lunroll] = convert_scheme_to_xdsl_transform_input(scheme, comp)

  assert(parallel_dims == ['n', 'h', 'w', 'f'])
  assert(reduction_dims == ['c', 'r', 's'])
  assert(dtiles_name == {'f': ['f1', 'f2'], 'x': ['x1', 'x2'], 'y': ['y1'], 'c': ['c1']})
  assert(ltilesizes == [('f2', 16), ('f1', 64), ('x2', 2), ('y1', 2), ('c1', 4), ('x1', 14)])
  assert(llooporder == ['h', 'w', 'x', 'y', 'x1', 'f', 'c', 'c1', 'y1', 'x2', 'f1', 'f2'])
  assert(ldim_vectorize == ['f2'])
  assert(l_dim_parallelized == [])
  assert(lunroll == [('f1', 4), ('x2', 2), ('y1', 2), ('c1', 4)])

  return

def test_convert_scheme_to_xdsl_transform_input_2():
  comp = Computation(Computation_spec.CONV, 4)
  str_scheme = "[V (F, 16); U (F, 4); U (X, 2); U (Y, 2); U (C, 4); T (C, 16); (Hoist_var([C])); (T (F, 2)); (T (X, 7)); (T (Y, 14)); (T (X, 2));  (T (W, 3)); (Tparal (H, 3))]"

  scheme = build_scheme_from_str(str_scheme)
  [parallel_dims, reduction_dims, dtiles_name,
    ltilesizes, llooporder, ldim_vectorize, l_dim_parallelized,
    lunroll] = convert_scheme_to_xdsl_transform_input(scheme, comp)
  
  assert(parallel_dims == ['n', 'h', 'w', 'f'])
  assert(reduction_dims == ['c', 'r', 's'])
  assert(dtiles_name == {'f': ['f1', 'f2'], 'x': ['x1', 'x2'], 'y': ['y1'], 'c': ['c1']})
  assert(ltilesizes == [('f2', 16), ('f1', 64), ('x2', 2), ('y1', 2), ('c1', 4), ('x1', 14)])
  assert(llooporder == ['h', 'w', 'x', 'y', 'x1', 'f', 'c', 'c1', 'y1', 'x2', 'f1', 'f2'])
  assert(ldim_vectorize == ['f2'])
  assert(l_dim_parallelized == ['h'])
  assert(lunroll == [('f1', 4), ('x2', 2), ('y1', 2), ('c1', 4)])

  return

def test_convert_scheme_to_xdsl_transform_input_3():
  comp = Computation(Computation_spec.CONV, 4)
  str_scheme = "[V(f,16); U(h,3); U(w,13); T(c,512)]"

  scheme = build_scheme_from_str(str_scheme)
  [parallel_dims, reduction_dims, dtiles_name,
    ltilesizes, llooporder, ldim_vectorize, l_dim_parallelized,
    lunroll] = convert_scheme_to_xdsl_transform_input(scheme, comp)

  assert(parallel_dims == ['n', 'h', 'w', 'f'])
  assert(reduction_dims == ['c', 'r', 's'])
  assert(dtiles_name == {})
  assert(ltilesizes == [])
  assert(llooporder == ['c', 'w', 'h', 'f'])
  assert(ldim_vectorize == ['f'])
  assert(l_dim_parallelized == [])
  assert(lunroll == [('h', 3), ('w', 13)])

  return


# Test of "convert_scheme_to_loopschedule"
def test_convert_scheme_to_loopschedule():
  comp = Computation(Computation_spec.MATMULT, 4)
  
  str_scheme = "[V (J, 16); U (J, 2); U (I, 8); T (K, 1024); T (J, 128); T (I, 32); T (J, 2); T (K, 2)]"
  scheme = build_scheme_from_str(str_scheme)

  machine = pinocchio_machine

  str_loopschedule = convert_scheme_to_loopschedule(comp, machine, scheme)
  assert(str_loopschedule=='"k", "j", "i", "j#4096", "k#1024", "i#8" = {"unroll"}, "j#32" = {"unroll"}, "j#16" = {"vectorize"}')

  return


# Test of "build_xdsl_module_string_matmul"
def test_build_xdsl_module_string_matmul():
  comp = Computation(Computation_spec.MATMULT, 4)
  
  dsizes = { "i" : 256, "j": 8192, "k": 2048 }
  str_scheme = "[V (J, 16); U (J, 2); U (I, 8); T (K, 1024); T (J, 128); T (I, 32); T (J, 2); T (K, 2)]"
  scheme = build_scheme_from_str(str_scheme)

  machine = pinocchio_machine

  str_mlir_out = build_xdsl_module_string_matmul(comp, machine, scheme, dsizes)
  expected_output = """func.func @myfun(
  %A: memref<256x2048xf32>,
  %B: memref<2048x8192xf32>,
  %C: memref<256x8192xf32>
) {
  linalg.matmul
    {
      loop.dims = ["i","j","k"],
      loop.schedule = {"k", "j", "i", "j#4096", "k#1024", "i#8" = {"unroll"}, "j#32" = {"unroll"}, "j#16" = {"vectorize"}}
    }
    ins(%A, %B : memref<256x2048xf32>, memref<2048x8192xf32>)
    outs(%C : memref<256x8192xf32>)
  return
}
"""
  assert(str_mlir_out==expected_output)
  return


# Test of "build_xdsl_module_string_conv"
def test_build_xdsl_module_string_conv():
  comp = Computation(Computation_spec.CONV, 4)
  
  dsizes = {"f" : 128, "c": 64, "h": 28, "w": 28, "r": 3, "s": 3, "strx": 1, "stry": 1}
  str_scheme = "[V (F, 16); U (F, 4); U (X, 2); U (Y, 2); U (C, 4); T (C, 16); (Hoist_var([C])); (T (F, 2)); (T (X, 7)); (T (Y, 14)); (T (X, 2));  (T (W, 3)); (T (H, 3))]"
  scheme = build_scheme_from_str(str_scheme)

  machine = pinocchio_machine

  str_mlir_out = build_xdsl_module_string_conv(comp, machine, scheme, dsizes)
  expected_output="""func.func @myfun(
  %I: memref<1x3x3x64xf32>,
  %K: memref<3x3x64x128xf32>,
  %O: memref<1x1x1x128xf32>
) {
  linalg.generic {
      indexing_maps = [
        affine_map<(n,h,w,f,r,s,c) -> (n,h+r,w+s,c)>,
        affine_map<(n,h,w,f,r,s,c) -> (r,s,c,f)>,
        affine_map<(n,h,w,f,r,s,c) -> (n,h,w,f)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel",
          "reduction", "reduction", "reduction"]
    }
    ins(%I, %K : memref<1x3x3x64xf32>, memref<3x3x64x128xf32>)
    outs(%O : memref<1x1x1x128xf32>)
     attrs = {
      loop.dims = ["n","h","w","f","r","s","c"],
      loop.schedule = {"r", "s", "h", "w", "h#14", "f", "c", "c#4" = {"unroll"}, "w#2" = {"unroll"}, "h#2" = {"unroll"}, "f#64" = {"unroll"}, "f#16" = {"vectorize"}}
     }
  {
    ^bb0(%0: f32, %1: f32, %2: f32) :
      %3 = arith.mulf %0, %1 : f32
      %4 = arith.addf %2, %3 : f32
      linalg.yield %4 : f32
  }
  return
}
"""
  assert(str_mlir_out == expected_output)

  return

# Test of "launch_and_measure_scheme_matmul"
# WARNING - actual execution on a machine. This was done on "laptop_guillaume_machine"
def test_launch_and_measure_scheme_matmul():
  comp = Computation(Computation_spec.MATMULT, 4)
  dsizes = { "i" : 8, "j": 32, "k": 64 }
  str_scheme = "[V (J, 16); U (J, 2); U (I, 8); T (K, 64); T(I, 1); T(J, 1)]"
  # Note: T(I, 1) + T(J, 1) due to a current limitation of xtc, that forbid a dimension to be fully unrolled
  #   (a tile which is not the full problem must be the one to be unrolled). Thus, this "hacky" workaround.
  scheme = build_scheme_from_str(str_scheme)
  machine = laptop_guillaume_machine

  res_measurement = launch_and_measure_scheme(comp, machine, scheme, dsizes)

  assert("peak_perf" in res_measurement.keys())

  return


# Test of "launch_and_measure_scheme_conv"
# WARNING - actual execution on a machine. This was done on "laptop_guillaume_machine"
def test_launch_and_measure_scheme_conv():
  comp = Computation(Computation_spec.CONV, 4)  # f32
  dsizes = {"f" : 64, "c": 32, "h": 1, "w": 1, "r": 1, "s": 1, "strx": 1, "stry": 1}
  str_scheme = "[V (F, 16); U (F, 2); U (C, 4); T (C, 16); (T (F, 2))]"
  scheme = build_scheme_from_str(str_scheme)
  machine = laptop_guillaume_machine

  res_measurement = launch_and_measure_scheme(comp, machine, scheme, dsizes)

  assert("peak_perf" in res_measurement.keys())

  return


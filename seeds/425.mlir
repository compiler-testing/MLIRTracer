module {
  func.func @main(%arg0: tensor<87x48x74x76xf32>, %arg1: tensor<19xi32>, %arg2: tensor<1xi32>) -> (tensor<1x48x74x76xf32>, tensor<1x48x74x2xi1>, tensor<1xi1>) {
    %0 = tosa.clamp %arg0 {min_val = 6.100000e+01 : f32, max_val = 1.230000e+02 : f32} : (tensor<87x48x74x76xf32>) -> tensor<87x48x74x76xf32>
    %1 = tosa.ceil %0 : (tensor<87x48x74x76xf32>) -> tensor<87x48x74x76xf32>
    %2 = tosa.reduce_sum %1 {axis = 0 : i32} : (tensor<87x48x74x76xf32>) -> tensor<1x48x74x76xf32>
    %3 = tosa.identity %2 : (tensor<1x48x74x76xf32>) -> tensor<1x48x74x76xf32>
    %4 = tosa.intdiv %arg1, %arg2 : (tensor<19xi32>, tensor<1xi32>) -> tensor<19xi32>
    %in_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %5 = tosa.negate %4, %in_zp_5, %out_zp_5 : (tensor<19xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<19xi32>
    %6 = tosa.reduce_min %3 {axis = 3 : i32} : (tensor<1x48x74x76xf32>) -> tensor<1x48x74x1xf32>
    %7 = tosa.greater %6, %6 : (tensor<1x48x74x1xf32>, tensor<1x48x74x1xf32>) -> tensor<1x48x74x1xi1>
    %8 = tosa.concat %7, %7 {axis = 3 : i32} : (tensor<1x48x74x1xi1>, tensor<1x48x74x1xi1>) -> tensor<1x48x74x2xi1>
    %9 = tosa.reduce_min %5 {axis = 0 : i32} : (tensor<19xi32>) -> tensor<1xi32>
    %10 = tosa.ceil %3 : (tensor<1x48x74x76xf32>) -> tensor<1x48x74x76xf32>
    %11 = tosa.logical_and %8, %8 : (tensor<1x48x74x2xi1>, tensor<1x48x74x2xi1>) -> tensor<1x48x74x2xi1>
    %12 = tosa.maximum %9, %9 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %13 = tosa.logical_left_shift %12, %12 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %14 = tosa.greater_equal %13, %12 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %15 = tosa.arithmetic_right_shift %14, %14 {round = false} : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    return %10, %11, %15 : tensor<1x48x74x76xf32>, tensor<1x48x74x2xi1>, tensor<1xi1>
  }
}

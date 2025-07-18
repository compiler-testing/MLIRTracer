module {
  func.func @main(%arg0: tensor<49xf32>, %arg1: tensor<i1>, %arg2: tensor<i1>, %arg3: tensor<i32>, %arg4: tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<49xf32>, tensor<49xf32>, tensor<i1>) {
    %0 = tosa.tanh %arg0 : (tensor<49xf32>) -> tensor<49xf32>
    %1 = tosa.logical_and %arg1, %arg2 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %2 = tosa.add %1, %1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %3 = tosa.intdiv %arg3, %arg4 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %4 = tosa.concat %0, %0 {axis = 0 : i32} : (tensor<49xf32>, tensor<49xf32>) -> tensor<98xf32>
    %5 = tosa.argmax %4 {axis = 0 : i32} : (tensor<98xf32>) -> tensor<i32>
    %6 = tosa.intdiv %5, %5 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %7 = tosa.logical_not %2 : (tensor<i1>) -> tensor<i1>
    %8 = tosa.logical_and %7, %1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %9 = tosa.logical_or %8, %8 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %in_zp_10 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_10 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %10 = tosa.negate %9, %in_zp_10, %out_zp_10 : (tensor<i1>, tensor<1xi1>, tensor<1xi1>) -> tensor<i1>
    %11 = tosa.rsqrt %0 : (tensor<49xf32>) -> tensor<49xf32>
    %12 = tosa.sigmoid %0 : (tensor<49xf32>) -> tensor<49xf32>
    %13 = tosa.logical_left_shift %10, %10 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %14 = tosa.logical_right_shift %2, %13 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    return %3, %6, %11, %12, %14 : tensor<i32>, tensor<i32>, tensor<49xf32>, tensor<49xf32>, tensor<i1>
  }
}

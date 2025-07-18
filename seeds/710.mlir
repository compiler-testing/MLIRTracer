module {
  func.func @main(%arg0: tensor<i32>, %arg1: tensor<44x67xi1>, %arg2: tensor<f32>) -> (tensor<f32>, tensor<i32>, tensor<1x1xi1>) {
    %in_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<i32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
    %1 = tosa.reduce_all %arg1 {axis = 0 : i32} : (tensor<44x67xi1>) -> tensor<1x67xi1>
    %2 = tosa.logical_right_shift %0, %0 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %3 = tosa.tanh %arg2 : (tensor<f32>) -> tensor<f32>
    %4 = tosa.rsqrt %3 : (tensor<f32>) -> tensor<f32>
    %5 = tosa.intdiv %2, %2 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %6 = tosa.reduce_all %1 {axis = 1 : i32} : (tensor<1x67xi1>) -> tensor<1x1xi1>
    return %4, %5, %6 : tensor<f32>, tensor<i32>, tensor<1x1xi1>
  }
}

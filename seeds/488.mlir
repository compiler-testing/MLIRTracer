module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i1>, %arg3: tensor<i1>, %arg4: tensor<76xf32>) -> (tensor<f32>, tensor<i1>, tensor<1xi1>) {
    %0 = tosa.pow %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %in_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.negate %0, %in_zp_1, %out_zp_1 : (tensor<f32>, tensor<1xf32>, tensor<1xf32>) -> tensor<f32>
    %2 = tosa.arithmetic_right_shift %arg2, %arg3 {round = true} : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %3 = tosa.reduce_min %arg4 {axis = 0 : i32} : (tensor<76xf32>) -> tensor<1xf32>
    %4 = tosa.greater_equal %3, %3 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %5 = tosa.bitwise_xor %4, %4 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    return %1, %2, %5 : tensor<f32>, tensor<i1>, tensor<1xi1>
  }
}

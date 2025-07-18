module {
  func.func @main(%arg0: tensor<i1>, %arg1: tensor<32x46x47x83xf32>, %arg2: tensor<52x17x2x93xf32>, %arg3: tensor<52xf32>) -> (tensor<32x111x97x52xf32>, tensor<i1>) {
    %0 = tosa.logical_not %arg0 : (tensor<i1>) -> tensor<i1>
    %1 = tosa.bitwise_or %0, %0 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %input_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.transpose_conv2d %arg1, %arg2, %arg3, %input_zp_2, %weight_zp_2 {acc_type = f32, out_pad = array<i64: 2, 2, 1, 2>, stride = array<i64: 2, 2>, out_shape = array<i64: 32, 111, 97, 52>} : (tensor<32x46x47x83xf32>, tensor<52x17x2x93xf32>, tensor<52xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<32x111x97x52xf32>
    %3 = tosa.arithmetic_right_shift %1, %1 {round = true} : (tensor<i1>, tensor<i1>) -> tensor<i1>
    return %2, %3 : tensor<32x111x97x52xf32>, tensor<i1>
  }
}

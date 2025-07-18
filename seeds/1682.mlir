module {
  func.func @main(%arg0: tensor<3x15x41x33xf32>, %arg1: tensor<36x28x13x4xf32>, %arg2: tensor<36xf32>, %arg3: tensor<60x10xi1>) -> (tensor<3x46x96x36xf32>, tensor<1x10xi1>) {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 2, 2, 2, 1>, stride = array<i64: 1, 2>, out_shape = array<i64: 3, 46, 96, 36>} : (tensor<3x15x41x33xf32>, tensor<36x28x13x4xf32>, tensor<36xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3x46x96x36xf32>
    %1 = tosa.rsqrt %0 : (tensor<3x46x96x36xf32>) -> tensor<3x46x96x36xf32>
    %2 = tosa.reduce_all %arg3 {axis = 0 : i32} : (tensor<60x10xi1>) -> tensor<1x10xi1>
    return %1, %2 : tensor<3x46x96x36xf32>, tensor<1x10xi1>
  }
}

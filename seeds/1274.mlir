module {
  func.func @main(%arg0: tensor<4x64x60x83xf32>, %arg1: tensor<37x95x11x100xf32>, %arg2: tensor<37xf32>, %arg3: tensor<i32>, %arg4: tensor<i32>) -> (tensor<4x161x72x1xf32>, tensor<i32>) {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 1, 2, 1, 1>, stride = array<i64: 1, 1>, out_shape = array<i64: 4, 161, 72, 37>} : (tensor<4x64x60x83xf32>, tensor<37x95x11x100xf32>, tensor<37xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<4x161x72x37xf32>
    %1 = tosa.reduce_sum %0 {axis = 3 : i32} : (tensor<4x161x72x37xf32>) -> tensor<4x161x72x1xf32>
    %2 = tosa.logical_right_shift %arg3, %arg4 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %3 = tosa.bitwise_and %2, %2 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return %1, %3 : tensor<4x161x72x1xf32>, tensor<i32>
  }
}

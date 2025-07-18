module {
  func.func @main(%arg0: tensor<57x31x21x90xf32>, %arg1: tensor<42x40x47x2xf32>, %arg2: tensor<42xf32>) -> tensor<57x74x71x42xf32> {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 2, 2, 2, 2>, stride = array<i64: 1, 1>, out_shape = array<i64: 57, 74, 71, 42>} : (tensor<57x31x21x90xf32>, tensor<42x40x47x2xf32>, tensor<42xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<57x74x71x42xf32>
    return %0 : tensor<57x74x71x42xf32>
  }
}

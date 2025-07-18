module {
  func.func @main(%arg0: tensor<13x82x57x34xf32>, %arg1: tensor<26x2x77x12xf32>, %arg2: tensor<26xf32>) -> tensor<13x166x192x26xf32> {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 1, 1, 1, 2>, stride = array<i64: 2, 2>, out_shape = array<i64: 13, 166, 192, 26>} : (tensor<13x82x57x34xf32>, tensor<26x2x77x12xf32>, tensor<26xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<13x166x192x26xf32>
    %1 = tosa.sub %0, %0 : (tensor<13x166x192x26xf32>, tensor<13x166x192x26xf32>) -> tensor<13x166x192x26xf32>
    return %1 : tensor<13x166x192x26xf32>
  }
}

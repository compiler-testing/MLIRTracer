module {
  func.func @main(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<83x81x50x5xf32>, %arg3: tensor<1x81x50x1xf32>, %arg4: tensor<75x69x44x46xf32>, %arg5: tensor<75xf32>) -> (tensor<i64>, tensor<83x232x145x75xf32>) {
    %0 = tosa.sub %arg0, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %1 = tosa.pow %arg2, %arg3 : (tensor<83x81x50x5xf32>, tensor<1x81x50x1xf32>) -> tensor<83x81x50x5xf32>
    %input_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.transpose_conv2d %1, %arg4, %arg5, %input_zp_2, %weight_zp_2 {acc_type = f32, out_pad = array<i64: 1, 2, 1, 2>, stride = array<i64: 2, 2>, out_shape = array<i64: 83, 232, 145, 75>} : (tensor<83x81x50x5xf32>, tensor<75x69x44x46xf32>, tensor<75xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<83x232x145x75xf32>
    return %0, %2 : tensor<i64>, tensor<83x232x145x75xf32>
  }
}

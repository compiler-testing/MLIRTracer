module {
  func.func @main(%arg0: tensor<88x41x3x4xf32>, %arg1: tensor<26x85x63x81xf32>, %arg2: tensor<26xf32>) -> tensor<88x167x68x26xf32> {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 1, 1, 1, 2>, stride = array<i64: 2, 1>, out_shape = array<i64: 88, 167, 68, 26>} : (tensor<88x41x3x4xf32>, tensor<26x85x63x81xf32>, tensor<26xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<88x167x68x26xf32>
    return %0 : tensor<88x167x68x26xf32>
  }
}

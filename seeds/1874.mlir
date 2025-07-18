module {
  func.func @main(%arg0: tensor<28x92x87x62xf32>, %arg1: tensor<7x27x82x61xf32>, %arg2: tensor<7xf32>) -> tensor<28x212x171x7xf32> {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 1, 2, 2, 1>, stride = array<i64: 2, 1>, out_shape = array<i64: 28, 212, 171, 7>} : (tensor<28x92x87x62xf32>, tensor<7x27x82x61xf32>, tensor<7xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<28x212x171x7xf32>
    return %0 : tensor<28x212x171x7xf32>
  }
}

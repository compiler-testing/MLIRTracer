module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<56x28x72x19xf32>, %arg2: tensor<44x86x80x80xf32>, %arg3: tensor<44xf32>) -> (tensor<56x117x154x44xf32>, tensor<f32>) {
    %0 = tosa.sigmoid %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.tanh %0 : (tensor<f32>) -> tensor<f32>
    %2 = tosa.rsqrt %1 : (tensor<f32>) -> tensor<f32>
    %input_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.transpose_conv2d %arg1, %arg2, %arg3, %input_zp_3, %weight_zp_3 {acc_type = f32, out_pad = array<i64: 2, 2, 1, 2>, stride = array<i64: 1, 1>, out_shape = array<i64: 56, 117, 154, 44>} : (tensor<56x28x72x19xf32>, tensor<44x86x80x80xf32>, tensor<44xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<56x117x154x44xf32>
    %4 = tosa.add %3, %3 : (tensor<56x117x154x44xf32>, tensor<56x117x154x44xf32>) -> tensor<56x117x154x44xf32>
    %5 = tosa.reciprocal %2 : (tensor<f32>) -> tensor<f32>
    return %4, %5 : tensor<56x117x154x44xf32>, tensor<f32>
  }
}

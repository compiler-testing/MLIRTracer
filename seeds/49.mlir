module {
  func.func @main(%arg0: tensor<85x73x95x59xf32>, %arg1: tensor<90x94x11x75xf32>, %arg2: tensor<90xf32>) -> tensor<85x169x107x90xf32> {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 2, 1, 1, 1>, stride = array<i64: 1, 1>, out_shape = array<i64: 85, 169, 107, 90>} : (tensor<85x73x95x59xf32>, tensor<90x94x11x75xf32>, tensor<90xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<85x169x107x90xf32>
    return %0 : tensor<85x169x107x90xf32>
  }
}

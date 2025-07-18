module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<56x79x36x82xf32>, %arg2: tensor<68x11x55x32xf32>, %arg3: tensor<68xf32>) -> (tensor<f32>, tensor<56x92x128x68xf32>) {
    %0 = tosa.sigmoid %arg0 : (tensor<f32>) -> tensor<f32>
    %input_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.transpose_conv2d %arg1, %arg2, %arg3, %input_zp_1, %weight_zp_1 {acc_type = f32, out_pad = array<i64: 2, 1, 2, 1>, stride = array<i64: 1, 2>, out_shape = array<i64: 56, 92, 128, 68>} : (tensor<56x79x36x82xf32>, tensor<68x11x55x32xf32>, tensor<68xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<56x92x128x68xf32>
    %2 = tosa.tanh %1 : (tensor<56x92x128x68xf32>) -> tensor<56x92x128x68xf32>
    return %0, %2 : tensor<f32>, tensor<56x92x128x68xf32>
  }
}

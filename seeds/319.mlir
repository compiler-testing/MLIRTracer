module {
  func.func @main(%arg0: tensor<38x24x49x82xf32>, %arg1: tensor<33x61x100x11xf32>, %arg2: tensor<33xf32>, %arg3: tensor<34x46x68xi1>, %arg4: tensor<34x46x1xi1>) -> (tensor<38x88x150x33xf32>, tensor<34x46x68xi1>) {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 2, 2, 1, 1>, stride = array<i64: 1, 1>, out_shape = array<i64: 38, 88, 150, 33>} : (tensor<38x24x49x82xf32>, tensor<33x61x100x11xf32>, tensor<33xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<38x88x150x33xf32>
    %1 = tosa.logical_or %arg3, %arg4 : (tensor<34x46x68xi1>, tensor<34x46x1xi1>) -> tensor<34x46x68xi1>
    return %0, %1 : tensor<38x88x150x33xf32>, tensor<34x46x68xi1>
  }
}

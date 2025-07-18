module {
  func.func @main(%arg0: tensor<5x38x81x21xf32>, %arg1: tensor<69x86x16x81xf32>, %arg2: tensor<69xf32>, %arg3: tensor<14x88x22x11xi32>, %arg4: tensor<1x1x1x1xi32>) -> (tensor<5x125x98x69xf32>, tensor<14x88x22x11xi32>) {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, out_shape = array<i64: 5, 125, 98, 69>} : (tensor<5x38x81x21xf32>, tensor<69x86x16x81xf32>, tensor<69xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<5x125x98x69xf32>
    %1 = tosa.add %0, %0 : (tensor<5x125x98x69xf32>, tensor<5x125x98x69xf32>) -> tensor<5x125x98x69xf32>
    %2 = tosa.logical_left_shift %arg3, %arg4 : (tensor<14x88x22x11xi32>, tensor<1x1x1x1xi32>) -> tensor<14x88x22x11xi32>
    return %1, %2 : tensor<5x125x98x69xf32>, tensor<14x88x22x11xi32>
  }
}

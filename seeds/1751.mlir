module {
  func.func @main(%arg0: tensor<36x9x74x52x90xi8>, %arg1: tensor<36x9x1x1x90xi8>, %arg2: tensor<44x13x69x20x91x11xf32>, %arg3: tensor<69x15x51x83xf32>, %arg4: tensor<64x80x95x17xf32>, %arg5: tensor<64xf32>) -> (tensor<36x9x74x52x90xi1>, tensor<44x13x69x20x91x11xf32>, tensor<69x97x198x64xf32>) {
    %0 = tosa.greater %arg0, %arg1 : (tensor<36x9x74x52x90xi8>, tensor<36x9x1x1x90xi8>) -> tensor<36x9x74x52x90xi1>
    %1 = tosa.reciprocal %arg2 : (tensor<44x13x69x20x91x11xf32>) -> tensor<44x13x69x20x91x11xf32>
    %input_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.transpose_conv2d %arg3, %arg4, %arg5, %input_zp_2, %weight_zp_2 {acc_type = f32, out_pad = array<i64: 2, 1, 1, 2>, stride = array<i64: 1, 2>, out_shape = array<i64: 69, 97, 198, 64>} : (tensor<69x15x51x83xf32>, tensor<64x80x95x17xf32>, tensor<64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<69x97x198x64xf32>
    return %0, %1, %2 : tensor<36x9x74x52x90xi1>, tensor<44x13x69x20x91x11xf32>, tensor<69x97x198x64xf32>
  }
}

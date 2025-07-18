module {
  func.func @main(%arg0: tensor<13x18x96x97xf32>, %arg1: tensor<70x9x68x92xf32>, %arg2: tensor<70xf32>, %arg3: tensor<98x6x69x100x90xi8>, %arg4: tensor<1x6x69x100x1xi8>) -> (tensor<13x30x261x70xf32>, tensor<98x6x69x100x90xi8>, tensor<98x6x69x100x90xi1>) {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 2, 2, 1, 2>, stride = array<i64: 1, 2>, out_shape = array<i64: 13, 30, 261, 70>} : (tensor<13x18x96x97xf32>, tensor<70x9x68x92xf32>, tensor<70xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<13x30x261x70xf32>
    %1 = tosa.logical_left_shift %arg3, %arg4 : (tensor<98x6x69x100x90xi8>, tensor<1x6x69x100x1xi8>) -> tensor<98x6x69x100x90xi8>
    %2 = tosa.clz %1 : (tensor<98x6x69x100x90xi8>) -> tensor<98x6x69x100x90xi8>
    %3 = tosa.greater %1, %1 : (tensor<98x6x69x100x90xi8>, tensor<98x6x69x100x90xi8>) -> tensor<98x6x69x100x90xi1>
    return %0, %2, %3 : tensor<13x30x261x70xf32>, tensor<98x6x69x100x90xi8>, tensor<98x6x69x100x90xi1>
  }
}

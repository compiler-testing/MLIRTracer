module {
  func.func @main(%arg0: tensor<97x50x33xf32>, %arg1: tensor<69x9x52x48x14xi1>, %arg2: tensor<1x1x1x48x1xi1>, %arg3: tensor<25x98x100x91xf32>, %arg4: tensor<73x100x51x26xf32>, %arg5: tensor<73xf32>) -> (tensor<97x50x33xf32>, tensor<69x9x52x48x14xi1>, tensor<25x298x251x73xi1>) {
    %0 = tosa.reverse %arg0 {axis = 2 : i32} : (tensor<97x50x33xf32>) -> tensor<97x50x33xf32>
    %1 = tosa.logical_and %arg1, %arg2 : (tensor<69x9x52x48x14xi1>, tensor<1x1x1x48x1xi1>) -> tensor<69x9x52x48x14xi1>
    %input_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.transpose_conv2d %arg3, %arg4, %arg5, %input_zp_2, %weight_zp_2 {acc_type = f32, out_pad = array<i64: 2, 2, 1, 1>, stride = array<i64: 2, 2>, out_shape = array<i64: 25, 298, 251, 73>} : (tensor<25x98x100x91xf32>, tensor<73x100x51x26xf32>, tensor<73xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<25x298x251x73xf32>
    %3 = tosa.greater %2, %2 : (tensor<25x298x251x73xf32>, tensor<25x298x251x73xf32>) -> tensor<25x298x251x73xi1>
    return %0, %1, %3 : tensor<97x50x33xf32>, tensor<69x9x52x48x14xi1>, tensor<25x298x251x73xi1>
  }
}

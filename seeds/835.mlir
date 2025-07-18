module {
  func.func @main(%arg0: tensor<43x59xf32>, %arg1: tensor<48x10x88x48xf32>, %arg2: tensor<19x18x47x13xf32>, %arg3: tensor<19xf32>, %arg4: tensor<62x27x11x75xi1>) -> (tensor<48x30x225x19xf32>, tensor<59xi32>, tensor<1x27x11x75xi1>) {
    %0 = tosa.exp %arg0 : (tensor<43x59xf32>) -> tensor<43x59xf32>
    %input_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.transpose_conv2d %arg1, %arg2, %arg3, %input_zp_1, %weight_zp_1 {acc_type = f32, out_pad = array<i64: 1, 2, 2, 2>, stride = array<i64: 1, 2>, out_shape = array<i64: 48, 30, 225, 19>} : (tensor<48x10x88x48xf32>, tensor<19x18x47x13xf32>, tensor<19xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<48x30x225x19xf32>
    %2 = tosa.argmax %0 {axis = 0 : i32} : (tensor<43x59xf32>) -> tensor<59xi32>
    %3 = tosa.sub %1, %1 : (tensor<48x30x225x19xf32>, tensor<48x30x225x19xf32>) -> tensor<48x30x225x19xf32>
    %4 = tosa.bitwise_or %2, %2 : (tensor<59xi32>, tensor<59xi32>) -> tensor<59xi32>
    %5 = tosa.reduce_any %arg4 {axis = 0 : i32} : (tensor<62x27x11x75xi1>) -> tensor<1x27x11x75xi1>
    return %3, %4, %5 : tensor<48x30x225x19xf32>, tensor<59xi32>, tensor<1x27x11x75xi1>
  }
}

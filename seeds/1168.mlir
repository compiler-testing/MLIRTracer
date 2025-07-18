module {
  func.func @main(%arg0: tensor<9x57x75x20x91x19xi64>, %arg1: tensor<9x57x75x1x1x19xi64>, %arg2: tensor<81x78x4x40xf32>, %arg3: tensor<25x77x86x5xf32>, %arg4: tensor<25xf32>, %arg5: tensor<88x60x73x53xi1>) -> (tensor<9x57x75x20x91x19xi64>, tensor<1x73x53xi32>, tensor<162x157x95x25xf32>, tensor<88x1x73x53xi1>, tensor<81x157x95x25xf32>, tensor<88x1x73x53xi1>) {
    %0 = tosa.arithmetic_right_shift %arg0, %arg1 {round = false} : (tensor<9x57x75x20x91x19xi64>, tensor<9x57x75x1x1x19xi64>) -> tensor<9x57x75x20x91x19xi64>
    %input_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.transpose_conv2d %arg2, %arg3, %arg4, %input_zp_1, %weight_zp_1 {acc_type = f32, out_pad = array<i64: 1, 2, 1, 2>, stride = array<i64: 1, 2>, out_shape = array<i64: 81, 157, 95, 25>} : (tensor<81x78x4x40xf32>, tensor<25x77x86x5xf32>, tensor<25xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<81x157x95x25xf32>
    %2 = tosa.bitwise_not %0 : (tensor<9x57x75x20x91x19xi64>) -> tensor<9x57x75x20x91x19xi64>
    %3 = tosa.reduce_all %arg5 {axis = 1 : i32} : (tensor<88x60x73x53xi1>) -> tensor<88x1x73x53xi1>
    %4 = tosa.argmax %3 {axis = 0 : i32} : (tensor<88x1x73x53xi1>) -> tensor<1x73x53xi32>
    %5 = tosa.abs %4 : (tensor<1x73x53xi32>) -> tensor<1x73x53xi32>
    %6 = tosa.tanh %1 : (tensor<81x157x95x25xf32>) -> tensor<81x157x95x25xf32>
    %7 = tosa.concat %6, %6 {axis = 0 : i32} : (tensor<81x157x95x25xf32>, tensor<81x157x95x25xf32>) -> tensor<162x157x95x25xf32>
    %8 = tosa.clz %3 : (tensor<88x1x73x53xi1>) -> tensor<88x1x73x53xi1>
    %9 = tosa.ceil %1 : (tensor<81x157x95x25xf32>) -> tensor<81x157x95x25xf32>
    %10 = tosa.logical_not %3 : (tensor<88x1x73x53xi1>) -> tensor<88x1x73x53xi1>
    return %2, %5, %7, %8, %9, %10 : tensor<9x57x75x20x91x19xi64>, tensor<1x73x53xi32>, tensor<162x157x95x25xf32>, tensor<88x1x73x53xi1>, tensor<81x157x95x25xf32>, tensor<88x1x73x53xi1>
  }
}

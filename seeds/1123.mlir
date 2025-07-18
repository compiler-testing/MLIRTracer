module {
  func.func @main(%arg0: tensor<29x59x36x19x7xi64>, %arg1: tensor<29x59x36x1x1xi64>, %arg2: tensor<25xi64>, %arg3: tensor<100x80x39x25x72xi1>, %arg4: tensor<100x80x39x1x72xi1>, %arg5: tensor<45x45x37x93xf32>) -> (tensor<29x59x36x19x7xi64>, tensor<100x80x39x25x72xi1>, tensor<1x50x1xi64>, tensor<45x45x37x93xf32>) {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<29x59x36x19x7xi64>, tensor<29x59x36x1x1xi64>) -> tensor<29x59x36x19x7xi64>
    %t_1 = tosa.const_shape {values = dense<[ 2 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %1 = tosa.tile %arg2, %t_1 : (tensor<25xi64>, !tosa.shape<1>) -> tensor<50xi64>
    %2 = tosa.minimum %1, %1 : (tensor<50xi64>, tensor<50xi64>) -> tensor<50xi64>
    %3 = tosa.abs %0 : (tensor<29x59x36x19x7xi64>) -> tensor<29x59x36x19x7xi64>
    %4 = tosa.logical_and %arg3, %arg4 : (tensor<100x80x39x25x72xi1>, tensor<100x80x39x1x72xi1>) -> tensor<100x80x39x25x72xi1>
    %r_5 = tosa.const_shape {values = dense<[ 1, 50, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %5 = tosa.reshape %2, %r_5 : (tensor<50xi64>, !tosa.shape<3>) -> tensor<1x50x1xi64>
    %6 = tosa.rsqrt %arg5 : (tensor<45x45x37x93xf32>) -> tensor<45x45x37x93xf32>
    return %3, %4, %5, %6 : tensor<29x59x36x19x7xi64>, tensor<100x80x39x25x72xi1>, tensor<1x50x1xi64>, tensor<45x45x37x93xf32>
  }
}

module {
  func.func @main(%arg0: tensor<2x61xi1>, %arg1: tensor<19x77x40x32x39xf32>, %arg2: tensor<19x77x40x32x1xf32>) -> (tensor<19x77x40x32x39xi1>, tensor<24x1x1xi1>) {
    %s_0_start = tosa.const_shape {values = dense<[ 0, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_0_size = tosa.const_shape {values = dense<[ 8, 12 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<2x61xi1>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<8x12xi1>
    %1 = tosa.bitwise_or %0, %0 : (tensor<8x12xi1>, tensor<8x12xi1>) -> tensor<8x12xi1>
    %2 = tosa.clz %1 : (tensor<8x12xi1>) -> tensor<8x12xi1>
    %r_3 = tosa.const_shape {values = dense<[ 24, 2, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %3 = tosa.reshape %2, %r_3 : (tensor<8x12xi1>, !tosa.shape<3>) -> tensor<24x2x2xi1>
    %4 = tosa.greater %arg1, %arg2 : (tensor<19x77x40x32x39xf32>, tensor<19x77x40x32x1xf32>) -> tensor<19x77x40x32x39xi1>
    %5 = tosa.reduce_any %3 {axis = 2 : i32} : (tensor<24x2x2xi1>) -> tensor<24x2x1xi1>
    %6 = tosa.reduce_any %5 {axis = 1 : i32} : (tensor<24x2x1xi1>) -> tensor<24x1x1xi1>
    return %4, %6 : tensor<19x77x40x32x39xi1>, tensor<24x1x1xi1>
  }
}

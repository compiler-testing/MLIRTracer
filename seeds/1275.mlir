module {
  func.func @main(%arg0: tensor<93x39x11xi1>, %arg1: tensor<52x67xf32>) -> (tensor<93x1x11xi1>, tensor<3484x1xf32>) {
    %0 = tosa.logical_not %arg0 : (tensor<93x39x11xi1>) -> tensor<93x39x11xi1>
    %1 = tosa.exp %arg1 : (tensor<52x67xf32>) -> tensor<52x67xf32>
    %2 = tosa.arithmetic_right_shift %0, %0 {round = false} : (tensor<93x39x11xi1>, tensor<93x39x11xi1>) -> tensor<93x39x11xi1>
    %3 = tosa.reduce_max %2 {axis = 1 : i32} : (tensor<93x39x11xi1>) -> tensor<93x1x11xi1>
    %4 = tosa.rsqrt %1 : (tensor<52x67xf32>) -> tensor<52x67xf32>
    %r_5 = tosa.const_shape {values = dense<[ 3484, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %5 = tosa.reshape %4, %r_5 : (tensor<52x67xf32>, !tosa.shape<2>) -> tensor<3484x1xf32>
    %6 = tosa.reverse %5 {axis = 0 : i32} : (tensor<3484x1xf32>) -> tensor<3484x1xf32>
    return %3, %6 : tensor<93x1x11xi1>, tensor<3484x1xf32>
  }
}

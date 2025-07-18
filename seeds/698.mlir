module {
  func.func @main(%arg0: tensor<11x70x3x17xi8>, %arg1: tensor<55x91x46xf32>) -> (tensor<1x91x46xf32>, tensor<1x1x3x51xi1>, tensor<11x1x3x51xi8>, tensor<1x1x3x51xi1>) {
    %t_0 = tosa.const_shape {values = dense<[ 1, 1, 1, 3 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %0 = tosa.tile %arg0, %t_0 : (tensor<11x70x3x17xi8>, !tosa.shape<4>) -> tensor<11x70x3x51xi8>
    %1 = tosa.reduce_product %0 {axis = 1 : i32} : (tensor<11x70x3x51xi8>) -> tensor<11x1x3x51xi8>
    %2 = tosa.logical_right_shift %1, %1 : (tensor<11x1x3x51xi8>, tensor<11x1x3x51xi8>) -> tensor<11x1x3x51xi8>
    %3 = tosa.equal %2, %2 : (tensor<11x1x3x51xi8>, tensor<11x1x3x51xi8>) -> tensor<11x1x3x51xi1>
    %4 = tosa.abs %3 : (tensor<11x1x3x51xi1>) -> tensor<11x1x3x51xi1>
    %5 = tosa.reduce_all %4 {axis = 0 : i32} : (tensor<11x1x3x51xi1>) -> tensor<1x1x3x51xi1>
    %6 = tosa.sub %5, %5 : (tensor<1x1x3x51xi1>, tensor<1x1x3x51xi1>) -> tensor<1x1x3x51xi1>
    %7 = tosa.reduce_product %6 {axis = 1 : i32} : (tensor<1x1x3x51xi1>) -> tensor<1x1x3x51xi1>
    %8 = tosa.logical_xor %7, %5 : (tensor<1x1x3x51xi1>, tensor<1x1x3x51xi1>) -> tensor<1x1x3x51xi1>
    %9 = tosa.tanh %arg1 : (tensor<55x91x46xf32>) -> tensor<55x91x46xf32>
    %10 = tosa.sigmoid %9 : (tensor<55x91x46xf32>) -> tensor<55x91x46xf32>
    %t_11 = tosa.const_shape {values = dense<[ 2, 1, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %11 = tosa.tile %10, %t_11 : (tensor<55x91x46xf32>, !tosa.shape<3>) -> tensor<110x91x46xf32>
    %12 = tosa.reduce_sum %11 {axis = 0 : i32} : (tensor<110x91x46xf32>) -> tensor<1x91x46xf32>
    %13 = tosa.logical_xor %7, %8 : (tensor<1x1x3x51xi1>, tensor<1x1x3x51xi1>) -> tensor<1x1x3x51xi1>
    %14 = tosa.arithmetic_right_shift %2, %1 {round = true} : (tensor<11x1x3x51xi8>, tensor<11x1x3x51xi8>) -> tensor<11x1x3x51xi8>
    %15 = tosa.bitwise_not %14 : (tensor<11x1x3x51xi8>) -> tensor<11x1x3x51xi8>
    %16 = tosa.logical_right_shift %6, %6 : (tensor<1x1x3x51xi1>, tensor<1x1x3x51xi1>) -> tensor<1x1x3x51xi1>
    return %12, %13, %15, %16 : tensor<1x91x46xf32>, tensor<1x1x3x51xi1>, tensor<11x1x3x51xi8>, tensor<1x1x3x51xi1>
  }
}

module {
  func.func @main(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<83x57x98xi1>) -> (tensor<i32>, tensor<1x57x98xi1>, tensor<1x19x1xi1>) {
    %0 = tosa.intdiv %arg0, %arg1 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %1 = tosa.reduce_any %arg2 {axis = 0 : i32} : (tensor<83x57x98xi1>) -> tensor<1x57x98xi1>
    %2 = tosa.abs %0 : (tensor<i32>) -> tensor<i32>
    %3 = tosa.reduce_any %1 {axis = 0 : i32} : (tensor<1x57x98xi1>) -> tensor<1x57x98xi1>
    %4 = tosa.logical_right_shift %3, %1 : (tensor<1x57x98xi1>, tensor<1x57x98xi1>) -> tensor<1x57x98xi1>
    %5 = tosa.reduce_max %1 {axis = 2 : i32} : (tensor<1x57x98xi1>) -> tensor<1x57x1xi1>
    %6 = tosa.logical_or %5, %5 : (tensor<1x57x1xi1>, tensor<1x57x1xi1>) -> tensor<1x57x1xi1>
    %7 = tosa.sub %6, %5 : (tensor<1x57x1xi1>, tensor<1x57x1xi1>) -> tensor<1x57x1xi1>
    %8 = tosa.bitwise_xor %4, %4 : (tensor<1x57x98xi1>, tensor<1x57x98xi1>) -> tensor<1x57x98xi1>
    %r_9 = tosa.const_shape {values = dense<[ 3, 19, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %9 = tosa.reshape %7, %r_9 : (tensor<1x57x1xi1>, !tosa.shape<3>) -> tensor<3x19x1xi1>
    %10 = tosa.bitwise_not %9 : (tensor<3x19x1xi1>) -> tensor<3x19x1xi1>
    %11 = tosa.logical_right_shift %8, %8 : (tensor<1x57x98xi1>, tensor<1x57x98xi1>) -> tensor<1x57x98xi1>
    %12 = tosa.reduce_product %10 {axis = 0 : i32} : (tensor<3x19x1xi1>) -> tensor<1x19x1xi1>
    return %2, %11, %12 : tensor<i32>, tensor<1x57x98xi1>, tensor<1x19x1xi1>
  }
}

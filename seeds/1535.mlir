module {
  func.func @main(%arg0: tensor<53x69x78x8x76xi1>, %arg1: tensor<72xi32>) -> (tensor<53x69x78x8x76xi1>, tensor<216xi32>) {
    %0 = tosa.bitwise_not %arg0 : (tensor<53x69x78x8x76xi1>) -> tensor<53x69x78x8x76xi1>
    %1 = tosa.arithmetic_right_shift %0, %0 {round = true} : (tensor<53x69x78x8x76xi1>, tensor<53x69x78x8x76xi1>) -> tensor<53x69x78x8x76xi1>
    %2 = tosa.clz %1 : (tensor<53x69x78x8x76xi1>) -> tensor<53x69x78x8x76xi1>
    %3 = tosa.logical_xor %2, %1 : (tensor<53x69x78x8x76xi1>, tensor<53x69x78x8x76xi1>) -> tensor<53x69x78x8x76xi1>
    %t_4 = tosa.const_shape {values = dense<[ 3 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %4 = tosa.tile %arg1, %t_4 : (tensor<72xi32>, !tosa.shape<1>) -> tensor<216xi32>
    %5 = tosa.intdiv %4, %4 : (tensor<216xi32>, tensor<216xi32>) -> tensor<216xi32>
    return %3, %5 : tensor<53x69x78x8x76xi1>, tensor<216xi32>
  }
}

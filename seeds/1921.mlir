module {
  func.func @main(%arg0: tensor<32x34xi8>) -> (tensor<1x34xi1>, tensor<11x8xi1>) {
    %0 = tosa.reduce_max %arg0 {axis = 0 : i32} : (tensor<32x34xi8>) -> tensor<1x34xi8>
    %1 = tosa.greater_equal %0, %0 : (tensor<1x34xi8>, tensor<1x34xi8>) -> tensor<1x34xi1>
    %2 = tosa.abs %1 : (tensor<1x34xi1>) -> tensor<1x34xi1>
    %3 = tosa.logical_right_shift %2, %2 : (tensor<1x34xi1>, tensor<1x34xi1>) -> tensor<1x34xi1>
    %4 = tosa.greater_equal %0, %0 : (tensor<1x34xi8>, tensor<1x34xi8>) -> tensor<1x34xi1>
    %s_5_start = tosa.const_shape {values = dense<[ 0, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_5_size = tosa.const_shape {values = dense<[ 11, 8 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %5 = tosa.slice %3, %s_5_start, %s_5_size : (tensor<1x34xi1>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<11x8xi1>
    return %4, %5 : tensor<1x34xi1>, tensor<11x8xi1>
  }
}

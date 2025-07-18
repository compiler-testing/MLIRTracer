module {
  func.func @main(%arg0: tensor<80xi1>, %arg1: tensor<24xf32>, %arg2: tensor<1xf32>) -> (tensor<1x1xi1>, tensor<1xf32>) {
    %0 = tosa.reduce_max %arg0 {axis = 0 : i32} : (tensor<80xi1>) -> tensor<1xi1>
    %r_1 = tosa.const_shape {values = dense<[ 1, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = tosa.reshape %0, %r_1 : (tensor<1xi1>, !tosa.shape<2>) -> tensor<1x1xi1>
    %2 = tosa.logical_or %1, %1 : (tensor<1x1xi1>, tensor<1x1xi1>) -> tensor<1x1xi1>
    %3 = tosa.logical_left_shift %2, %2 : (tensor<1x1xi1>, tensor<1x1xi1>) -> tensor<1x1xi1>
    %4 = tosa.pow %arg1, %arg2 : (tensor<24xf32>, tensor<1xf32>) -> tensor<24xf32>
    %s_5_start = tosa.const_shape {values = dense<[ 8 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_5_size = tosa.const_shape {values = dense<[ 1 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %5 = tosa.slice %4, %s_5_start, %s_5_size : (tensor<24xf32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<1xf32>
    return %3, %5 : tensor<1x1xi1>, tensor<1xf32>
  }
}

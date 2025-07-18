module {
  func.func @main(%arg0: tensor<17xi32>) -> tensor<1xi32> {
    %s_0_start = tosa.const_shape {values = dense<[ 2 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_0_size = tosa.const_shape {values = dense<[ 12 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<17xi32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<12xi32>
    %1 = tosa.logical_right_shift %0, %0 : (tensor<12xi32>, tensor<12xi32>) -> tensor<12xi32>
    %2 = tosa.maximum %1, %1 : (tensor<12xi32>, tensor<12xi32>) -> tensor<12xi32>
    %3 = tosa.add %2, %1 : (tensor<12xi32>, tensor<12xi32>) -> tensor<12xi32>
    %4 = tosa.bitwise_xor %3, %1 : (tensor<12xi32>, tensor<12xi32>) -> tensor<12xi32>
    %5 = tosa.reduce_min %4 {axis = 0 : i32} : (tensor<12xi32>) -> tensor<1xi32>
    %6 = tosa.clz %5 : (tensor<1xi32>) -> tensor<1xi32>
    %7 = tosa.reduce_max %6 {axis = 0 : i32} : (tensor<1xi32>) -> tensor<1xi32>
    %8 = tosa.abs %7 : (tensor<1xi32>) -> tensor<1xi32>
    return %8 : tensor<1xi32>
  }
}

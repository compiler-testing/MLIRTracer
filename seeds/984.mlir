module {
  func.func @main(%arg0: tensor<85xf32>, %arg1: tensor<50x15x73xi1>, %arg2: tensor<50x15x1xi1>) -> (tensor<50x15x73xi1>, tensor<4xf32>) {
    %s_0_start = tosa.const_shape {values = dense<[ 6 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_0_size = tosa.const_shape {values = dense<[ 4 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<85xf32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<4xf32>
    %1 = tosa.identity %0 : (tensor<4xf32>) -> tensor<4xf32>
    %2 = tosa.logical_xor %arg1, %arg2 : (tensor<50x15x73xi1>, tensor<50x15x1xi1>) -> tensor<50x15x73xi1>
    %3 = tosa.bitwise_or %2, %2 : (tensor<50x15x73xi1>, tensor<50x15x73xi1>) -> tensor<50x15x73xi1>
    %4 = tosa.sigmoid %1 : (tensor<4xf32>) -> tensor<4xf32>
    return %3, %4 : tensor<50x15x73xi1>, tensor<4xf32>
  }
}

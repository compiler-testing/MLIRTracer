module {
  func.func @main(%arg0: tensor<59xi32>, %arg1: tensor<27x92x63x99xi1>, %arg2: tensor<27x92x1x99xi1>) -> (tensor<27x92x63x99xi1>, tensor<1xi32>) {
    %s_0_start = tosa.const_shape {values = dense<[ 4 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_0_size = tosa.const_shape {values = dense<[ 8 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<59xi32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<8xi32>
    %1 = tosa.bitwise_or %0, %0 : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
    %2 = tosa.logical_and %arg1, %arg2 : (tensor<27x92x63x99xi1>, tensor<27x92x1x99xi1>) -> tensor<27x92x63x99xi1>
    %s_3_start = tosa.const_shape {values = dense<[ 7 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_3_size = tosa.const_shape {values = dense<[ 1 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %3 = tosa.slice %1, %s_3_start, %s_3_size : (tensor<8xi32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<1xi32>
    return %2, %3 : tensor<27x92x63x99xi1>, tensor<1xi32>
  }
}

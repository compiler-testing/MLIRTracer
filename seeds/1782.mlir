module {
  func.func @main(%arg0: tensor<29xf32>, %arg1: tensor<3xi1>) -> (tensor<29xf32>, tensor<2xi1>) {
    %0 = tosa.ceil %arg0 : (tensor<29xf32>) -> tensor<29xf32>
    %1 = tosa.logical_not %arg1 : (tensor<3xi1>) -> tensor<3xi1>
    %s_2_start = tosa.const_shape {values = dense<[ 1 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_2_size = tosa.const_shape {values = dense<[ 2 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %2 = tosa.slice %1, %s_2_start, %s_2_size : (tensor<3xi1>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<2xi1>
    %3 = tosa.bitwise_and %2, %2 : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
    return %0, %3 : tensor<29xf32>, tensor<2xi1>
  }
}

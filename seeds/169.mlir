module {
  func.func @main(%arg0: tensor<50x23x67x56x8xi32>, %arg1: tensor<77xi1>, %arg2: tensor<77xi1>) -> (tensor<50x23x67x56x8xi32>, tensor<6xi1>, tensor<1xi1>) {
    %0 = tosa.identity %arg0 : (tensor<50x23x67x56x8xi32>) -> tensor<50x23x67x56x8xi32>
    %1 = tosa.logical_xor %arg1, %arg2 : (tensor<77xi1>, tensor<77xi1>) -> tensor<77xi1>
    %s_2_start = tosa.const_shape {values = dense<[ 71 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_2_size = tosa.const_shape {values = dense<[ 6 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %2 = tosa.slice %1, %s_2_start, %s_2_size : (tensor<77xi1>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<6xi1>
    %3 = tosa.bitwise_and %2, %2 : (tensor<6xi1>, tensor<6xi1>) -> tensor<6xi1>
    %4 = tosa.logical_not %3 : (tensor<6xi1>) -> tensor<6xi1>
    %5 = tosa.reduce_min %3 {axis = 0 : i32} : (tensor<6xi1>) -> tensor<1xi1>
    return %0, %4, %5 : tensor<50x23x67x56x8xi32>, tensor<6xi1>, tensor<1xi1>
  }
}

module {
  func.func @main(%arg0: tensor<38xi1>, %arg1: tensor<38xi1>) -> tensor<1xi1> {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<38xi1>, tensor<38xi1>) -> tensor<38xi1>
    %1 = tosa.add %0, %0 : (tensor<38xi1>, tensor<38xi1>) -> tensor<38xi1>
    %2 = tosa.logical_right_shift %1, %1 : (tensor<38xi1>, tensor<38xi1>) -> tensor<38xi1>
    %t_3 = tosa.const_shape {values = dense<[ 2 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %3 = tosa.tile %2, %t_3 : (tensor<38xi1>, !tosa.shape<1>) -> tensor<76xi1>
    %4 = tosa.reduce_any %3 {axis = 0 : i32} : (tensor<76xi1>) -> tensor<1xi1>
    %5 = tosa.logical_not %4 : (tensor<1xi1>) -> tensor<1xi1>
    return %5 : tensor<1xi1>
  }
}

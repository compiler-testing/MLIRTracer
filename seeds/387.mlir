module {
  func.func @main(%arg0: tensor<14x26xi64>, %arg1: tensor<39x70xi1>) -> (tensor<14x78xi64>, tensor<39x1xi1>) {
    %t_0 = tosa.const_shape {values = dense<[ 1, 3 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %0 = tosa.tile %arg0, %t_0 : (tensor<14x26xi64>, !tosa.shape<2>) -> tensor<14x78xi64>
    %1 = tosa.reduce_any %arg1 {axis = 1 : i32} : (tensor<39x70xi1>) -> tensor<39x1xi1>
    %2 = tosa.logical_xor %1, %1 : (tensor<39x1xi1>, tensor<39x1xi1>) -> tensor<39x1xi1>
    %3 = tosa.logical_right_shift %2, %2 : (tensor<39x1xi1>, tensor<39x1xi1>) -> tensor<39x1xi1>
    return %0, %3 : tensor<14x78xi64>, tensor<39x1xi1>
  }
}

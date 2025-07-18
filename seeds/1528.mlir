module {
  func.func @main(%arg0: tensor<78x31x7xf32>, %arg1: tensor<73xi1>, %arg2: tensor<1xi1>) -> (tensor<73xi1>, tensor<156x31x21xi1>) {
    %t_0 = tosa.const_shape {values = dense<[ 2, 1, 3 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %0 = tosa.tile %arg0, %t_0 : (tensor<78x31x7xf32>, !tosa.shape<3>) -> tensor<156x31x21xf32>
    %1 = tosa.logical_or %arg1, %arg2 : (tensor<73xi1>, tensor<1xi1>) -> tensor<73xi1>
    %2 = tosa.equal %0, %0 : (tensor<156x31x21xf32>, tensor<156x31x21xf32>) -> tensor<156x31x21xi1>
    %3 = tosa.identity %1 : (tensor<73xi1>) -> tensor<73xi1>
    %4 = tosa.identity %2 : (tensor<156x31x21xi1>) -> tensor<156x31x21xi1>
    return %3, %4 : tensor<73xi1>, tensor<156x31x21xi1>
  }
}

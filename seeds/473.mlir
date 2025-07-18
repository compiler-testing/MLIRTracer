module {
  func.func @main(%arg0: tensor<56x98xi1>, %arg1: tensor<1x98xi1>) -> tensor<56x98xi1> {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<56x98xi1>, tensor<1x98xi1>) -> tensor<56x98xi1>
    return %0 : tensor<56x98xi1>
  }
}

module {
  func.func @main(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    return %0 : tensor<i1>
  }
}

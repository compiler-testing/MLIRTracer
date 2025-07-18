module {
  func.func @main(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %1 = tosa.bitwise_xor %0, %0 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    return %1 : tensor<i1>
  }
}

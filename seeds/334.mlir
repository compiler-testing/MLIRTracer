module {
  func.func @main(%arg0: tensor<i1>) -> tensor<i1> {
    %0 = tosa.bitwise_not %arg0 : (tensor<i1>) -> tensor<i1>
    %1 = tosa.clz %0 : (tensor<i1>) -> tensor<i1>
    return %1 : tensor<i1>
  }
}

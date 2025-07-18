module {
  func.func @main(%arg0: tensor<i1>) -> tensor<i1> {
    %0 = tosa.abs %arg0 : (tensor<i1>) -> tensor<i1>
    return %0 : tensor<i1>
  }
}

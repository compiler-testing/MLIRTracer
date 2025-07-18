module {
  func.func @main(%arg0: tensor<i8>, %arg1: tensor<i8>) -> tensor<i1> {
    %0 = tosa.equal %arg0, %arg1 : (tensor<i8>, tensor<i8>) -> tensor<i1>
    return %0 : tensor<i1>
  }
}

module {
  func.func @main(%arg0: tensor<i8>, %arg1: tensor<i8>) -> tensor<i8> {
    %0 = tosa.sub %arg0, %arg1 : (tensor<i8>, tensor<i8>) -> tensor<i8>
    return %0 : tensor<i8>
  }
}

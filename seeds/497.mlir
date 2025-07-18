module {
  func.func @main(%arg0: tensor<28xi64>) -> tensor<28xi64> {
    %0 = tosa.abs %arg0 : (tensor<28xi64>) -> tensor<28xi64>
    return %0 : tensor<28xi64>
  }
}

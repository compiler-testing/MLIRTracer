module {
  func.func @main(%arg0: tensor<82x35x30xi64>) -> tensor<82x35x30xi64> {
    %0 = tosa.abs %arg0 : (tensor<82x35x30xi64>) -> tensor<82x35x30xi64>
    return %0 : tensor<82x35x30xi64>
  }
}

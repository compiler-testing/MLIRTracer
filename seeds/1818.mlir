module {
  func.func @main(%arg0: tensor<1x83x5x35x28xi64>, %arg1: tensor<1x1x1x35x28xi64>) -> tensor<1x83x5x35x28xi64> {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<1x83x5x35x28xi64>, tensor<1x1x1x35x28xi64>) -> tensor<1x83x5x35x28xi64>
    return %0 : tensor<1x83x5x35x28xi64>
  }
}

module {
  func.func @main(%arg0: tensor<95x32x20xi64>, %arg1: tensor<1x1x1xi64>) -> tensor<95x32x20xi64> {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<95x32x20xi64>, tensor<1x1x1xi64>) -> tensor<95x32x20xi64>
    return %0 : tensor<95x32x20xi64>
  }
}

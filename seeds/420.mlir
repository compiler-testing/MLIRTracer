module {
  func.func @main(%arg0: tensor<55x43x2x7x12xi64>, %arg1: tensor<55x1x2x7x1xi64>) -> tensor<55x43x2x7x12xi64> {
    %0 = tosa.sub %arg0, %arg1 : (tensor<55x43x2x7x12xi64>, tensor<55x1x2x7x1xi64>) -> tensor<55x43x2x7x12xi64>
    return %0 : tensor<55x43x2x7x12xi64>
  }
}

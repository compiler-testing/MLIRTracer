module {
  func.func @main(%arg0: tensor<100x2xi64>, %arg1: tensor<100x2xi64>) -> tensor<100x2xi64> {
    %0 = tosa.bitwise_or %arg0, %arg1 : (tensor<100x2xi64>, tensor<100x2xi64>) -> tensor<100x2xi64>
    return %0 : tensor<100x2xi64>
  }
}

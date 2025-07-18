module {
  func.func @main(%arg0: tensor<45xi64>) -> tensor<45xi1> {
    %0 = tosa.bitwise_not %arg0 : (tensor<45xi64>) -> tensor<45xi64>
    %1 = tosa.greater %0, %0 : (tensor<45xi64>, tensor<45xi64>) -> tensor<45xi1>
    return %1 : tensor<45xi1>
  }
}

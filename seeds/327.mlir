module {
  func.func @main(%arg0: tensor<80x1xi64>, %arg1: tensor<80x1xi64>) -> tensor<80x1xi64> {
    %0 = tosa.sub %arg0, %arg1 : (tensor<80x1xi64>, tensor<80x1xi64>) -> tensor<80x1xi64>
    return %0 : tensor<80x1xi64>
  }
}

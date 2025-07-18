module {
  func.func @main(%arg0: tensor<18x3x44x100xi64>, %arg1: tensor<18x3x1x1xi64>) -> tensor<18x3x44x100xi64> {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<18x3x44x100xi64>, tensor<18x3x1x1xi64>) -> tensor<18x3x44x100xi64>
    return %0 : tensor<18x3x44x100xi64>
  }
}

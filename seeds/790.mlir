module {
  func.func @main(%arg0: tensor<34xi64>, %arg1: tensor<1xi64>) -> tensor<34xi64> {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<34xi64>, tensor<1xi64>) -> tensor<34xi64>
    %1 = tosa.identity %0 : (tensor<34xi64>) -> tensor<34xi64>
    return %1 : tensor<34xi64>
  }
}

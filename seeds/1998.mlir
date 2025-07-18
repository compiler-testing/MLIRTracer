module {
  func.func @main(%arg0: tensor<100xi64>, %arg1: tensor<1xi64>, %arg2: tensor<56x33xf32>) -> (tensor<100xi64>, tensor<56x33xf32>) {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<100xi64>, tensor<1xi64>) -> tensor<100xi64>
    %1 = tosa.log %arg2 : (tensor<56x33xf32>) -> tensor<56x33xf32>
    return %0, %1 : tensor<100xi64>, tensor<56x33xf32>
  }
}

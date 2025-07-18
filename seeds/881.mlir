module {
  func.func @main(%arg0: tensor<66x65x54x13xi64>, %arg1: tensor<1x65x54x13xi64>, %arg2: tensor<82x75xf32>) -> (tensor<82x75xf32>, tensor<66x65x54x13xi1>) {
    %0 = tosa.greater %arg0, %arg1 : (tensor<66x65x54x13xi64>, tensor<1x65x54x13xi64>) -> tensor<66x65x54x13xi1>
    %1 = tosa.ceil %arg2 : (tensor<82x75xf32>) -> tensor<82x75xf32>
    %2 = tosa.logical_right_shift %0, %0 : (tensor<66x65x54x13xi1>, tensor<66x65x54x13xi1>) -> tensor<66x65x54x13xi1>
    return %1, %2 : tensor<82x75xf32>, tensor<66x65x54x13xi1>
  }
}

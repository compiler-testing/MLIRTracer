module {
  func.func @main(%arg0: tensor<59x32x57x3xi64>, %arg1: tensor<1x32x57x1xi64>, %arg2: tensor<87x74xf32>) -> (tensor<59x32x57x3xi64>, tensor<87x74xf32>) {
    %0 = tosa.arithmetic_right_shift %arg0, %arg1 {round = true} : (tensor<59x32x57x3xi64>, tensor<1x32x57x1xi64>) -> tensor<59x32x57x3xi64>
    %1 = tosa.floor %arg2 : (tensor<87x74xf32>) -> tensor<87x74xf32>
    return %0, %1 : tensor<59x32x57x3xi64>, tensor<87x74xf32>
  }
}

module {
  func.func @main(%arg0: tensor<83x14x29x64xi64>, %arg1: tensor<1x14x29x64xi64>, %arg2: tensor<29xf32>) -> (tensor<83x14x29x64xi64>, tensor<29xf32>) {
    %0 = tosa.arithmetic_right_shift %arg0, %arg1 {round = true} : (tensor<83x14x29x64xi64>, tensor<1x14x29x64xi64>) -> tensor<83x14x29x64xi64>
    %1 = tosa.tanh %arg2 : (tensor<29xf32>) -> tensor<29xf32>
    return %0, %1 : tensor<83x14x29x64xi64>, tensor<29xf32>
  }
}

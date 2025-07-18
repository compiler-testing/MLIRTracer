module {
  func.func @main(%arg0: tensor<17x7x46x39x32x24xi64>, %arg1: tensor<17x7x1x39x32x1xi64>) -> tensor<17x7x46x39x32x24xi64> {
    %0 = tosa.arithmetic_right_shift %arg0, %arg1 {round = true} : (tensor<17x7x46x39x32x24xi64>, tensor<17x7x1x39x32x1xi64>) -> tensor<17x7x46x39x32x24xi64>
    return %0 : tensor<17x7x46x39x32x24xi64>
  }
}

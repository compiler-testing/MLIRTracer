module {
  func.func @main(%arg0: tensor<55x2x33xi16>, %arg1: tensor<55x2x1xi16>) -> tensor<55x2x33xi16> {
    %0 = tosa.arithmetic_right_shift %arg0, %arg1 {round = true} : (tensor<55x2x33xi16>, tensor<55x2x1xi16>) -> tensor<55x2x33xi16>
    return %0 : tensor<55x2x33xi16>
  }
}

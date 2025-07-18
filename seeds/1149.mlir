module {
  func.func @main(%arg0: tensor<75x32x78xi16>, %arg1: tensor<75x32x78xi16>) -> tensor<75x32x78xi16> {
    %0 = tosa.arithmetic_right_shift %arg0, %arg1 {round = true} : (tensor<75x32x78xi16>, tensor<75x32x78xi16>) -> tensor<75x32x78xi16>
    return %0 : tensor<75x32x78xi16>
  }
}

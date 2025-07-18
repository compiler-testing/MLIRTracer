module {
  func.func @main(%arg0: tensor<100x13x97x43x92xf32>, %arg1: tensor<63xi16>, %arg2: tensor<63xi16>) -> (tensor<100x13x97x43x92xf32>, tensor<63xi16>) {
    %0 = tosa.sigmoid %arg0 : (tensor<100x13x97x43x92xf32>) -> tensor<100x13x97x43x92xf32>
    %1 = tosa.arithmetic_right_shift %arg1, %arg2 {round = false} : (tensor<63xi16>, tensor<63xi16>) -> tensor<63xi16>
    return %0, %1 : tensor<100x13x97x43x92xf32>, tensor<63xi16>
  }
}

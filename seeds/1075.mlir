module {
  func.func @main(%arg0: tensor<70x81x33xi16>, %arg1: tensor<1x81x33xi16>, %arg2: tensor<i1>) -> (tensor<70x81x33xi16>, tensor<i1>) {
    %0 = tosa.logical_left_shift %arg0, %arg1 : (tensor<70x81x33xi16>, tensor<1x81x33xi16>) -> tensor<70x81x33xi16>
    %1 = tosa.logical_not %arg2 : (tensor<i1>) -> tensor<i1>
    return %0, %1 : tensor<70x81x33xi16>, tensor<i1>
  }
}

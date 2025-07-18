module {
  func.func @main(%arg0: tensor<53x62x7x93x21xi32>, %arg1: tensor<1x62x1x93x1xi32>, %arg2: tensor<66x28x13x24x59xi1>, %arg3: tensor<1x1x13x1x1xi1>) -> (tensor<53x62x7x93x21xi32>, tensor<66x28x13x24x59xi1>) {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<53x62x7x93x21xi32>, tensor<1x62x1x93x1xi32>) -> tensor<53x62x7x93x21xi32>
    %1 = tosa.sub %0, %0 : (tensor<53x62x7x93x21xi32>, tensor<53x62x7x93x21xi32>) -> tensor<53x62x7x93x21xi32>
    %2 = tosa.logical_or %arg2, %arg3 : (tensor<66x28x13x24x59xi1>, tensor<1x1x13x1x1xi1>) -> tensor<66x28x13x24x59xi1>
    return %1, %2 : tensor<53x62x7x93x21xi32>, tensor<66x28x13x24x59xi1>
  }
}

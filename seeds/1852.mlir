module {
  func.func @main(%arg0: tensor<76x5x13x30xi1>, %arg1: tensor<76x5x1x30xi1>) -> tensor<76x5x13x30xi1> {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<76x5x13x30xi1>, tensor<76x5x1x30xi1>) -> tensor<76x5x13x30xi1>
    %1 = tosa.logical_left_shift %0, %0 : (tensor<76x5x13x30xi1>, tensor<76x5x13x30xi1>) -> tensor<76x5x13x30xi1>
    %2 = tosa.logical_xor %1, %0 : (tensor<76x5x13x30xi1>, tensor<76x5x13x30xi1>) -> tensor<76x5x13x30xi1>
    return %2 : tensor<76x5x13x30xi1>
  }
}

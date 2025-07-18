module {
  func.func @main(%arg0: tensor<i16>, %arg1: tensor<i16>) -> tensor<i16> {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<i16>, tensor<i16>) -> tensor<i16>
    return %0 : tensor<i16>
  }
}

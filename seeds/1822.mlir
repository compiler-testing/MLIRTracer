module {
  func.func @main(%arg0: tensor<37x15x82x34x1x26xf32>, %arg1: tensor<i16>, %arg2: tensor<i16>) -> (tensor<i16>, tensor<37x15x82x34x1x26xf32>) {
    %0 = tosa.log %arg0 : (tensor<37x15x82x34x1x26xf32>) -> tensor<37x15x82x34x1x26xf32>
    %1 = tosa.bitwise_xor %arg1, %arg2 : (tensor<i16>, tensor<i16>) -> tensor<i16>
    %2 = tosa.bitwise_or %1, %1 : (tensor<i16>, tensor<i16>) -> tensor<i16>
    %3 = tosa.tanh %0 : (tensor<37x15x82x34x1x26xf32>) -> tensor<37x15x82x34x1x26xf32>
    return %2, %3 : tensor<i16>, tensor<37x15x82x34x1x26xf32>
  }
}

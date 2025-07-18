module {
  func.func @main(%arg0: tensor<i16>, %arg1: tensor<i16>, %arg2: tensor<82x7xf32>) -> (tensor<82x7xf32>, tensor<i16>) {
    %0 = tosa.arithmetic_right_shift %arg0, %arg1 {round = true} : (tensor<i16>, tensor<i16>) -> tensor<i16>
    %1 = tosa.ceil %arg2 : (tensor<82x7xf32>) -> tensor<82x7xf32>
    %2 = tosa.clz %0 : (tensor<i16>) -> tensor<i16>
    return %1, %2 : tensor<82x7xf32>, tensor<i16>
  }
}

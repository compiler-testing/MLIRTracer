module {
  func.func @main(%arg0: tensor<86x80x70x4x54x31xf32>, %arg1: tensor<i16>, %arg2: tensor<i16>) -> (tensor<86x80x70x4x54x31xf32>, tensor<i16>) {
    %0 = tosa.tanh %arg0 : (tensor<86x80x70x4x54x31xf32>) -> tensor<86x80x70x4x54x31xf32>
    %1 = tosa.bitwise_and %arg1, %arg2 : (tensor<i16>, tensor<i16>) -> tensor<i16>
    return %0, %1 : tensor<86x80x70x4x54x31xf32>, tensor<i16>
  }
}

module {
  func.func @main(%arg0: tensor<i16>, %arg1: tensor<i16>, %arg2: tensor<4x19x51x17x16xf32>, %arg3: tensor<4x19x51x17x16xf32>) -> (tensor<i16>, tensor<4x19x51x17x16xf32>) {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<i16>, tensor<i16>) -> tensor<i16>
    %1 = tosa.pow %arg2, %arg3 : (tensor<4x19x51x17x16xf32>, tensor<4x19x51x17x16xf32>) -> tensor<4x19x51x17x16xf32>
    %2 = tosa.reciprocal %1 : (tensor<4x19x51x17x16xf32>) -> tensor<4x19x51x17x16xf32>
    return %0, %2 : tensor<i16>, tensor<4x19x51x17x16xf32>
  }
}

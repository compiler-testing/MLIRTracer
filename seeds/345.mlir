module {
  func.func @main(%arg0: tensor<3x32x60x71x97x3xi16>, %arg1: tensor<1x1x1x1x97x3xi16>, %arg2: tensor<41x78x19x18xi32>, %arg3: tensor<1x78x19x18xi32>, %arg4: tensor<4x84x73x100x35x96xf32>) -> (tensor<3x32x60x71x97x3xi16>, tensor<41x78x19x18xi1>, tensor<4x84x73x100x35x96xf32>) {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<3x32x60x71x97x3xi16>, tensor<1x1x1x1x97x3xi16>) -> tensor<3x32x60x71x97x3xi16>
    %1 = tosa.greater_equal %arg2, %arg3 : (tensor<41x78x19x18xi32>, tensor<1x78x19x18xi32>) -> tensor<41x78x19x18xi1>
    %2 = tosa.bitwise_and %1, %1 : (tensor<41x78x19x18xi1>, tensor<41x78x19x18xi1>) -> tensor<41x78x19x18xi1>
    %3 = tosa.logical_right_shift %2, %1 : (tensor<41x78x19x18xi1>, tensor<41x78x19x18xi1>) -> tensor<41x78x19x18xi1>
    %4 = tosa.ceil %arg4 : (tensor<4x84x73x100x35x96xf32>) -> tensor<4x84x73x100x35x96xf32>
    %5 = tosa.logical_or %3, %3 : (tensor<41x78x19x18xi1>, tensor<41x78x19x18xi1>) -> tensor<41x78x19x18xi1>
    %6 = tosa.sigmoid %4 : (tensor<4x84x73x100x35x96xf32>) -> tensor<4x84x73x100x35x96xf32>
    return %0, %5, %6 : tensor<3x32x60x71x97x3xi16>, tensor<41x78x19x18xi1>, tensor<4x84x73x100x35x96xf32>
  }
}
